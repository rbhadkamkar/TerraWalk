/*
 * TerraWalk browser-side experience components.
 *
 * The application is intentionally kept as a Flask/Jinja + vanilla JavaScript
 * project. React is used only for small, isolated islands on the Landing,
 * Robot Systems, and Pipeline views; the Three.js simulation DOM is never
 * mounted, read, or modified by this file.
 *
 * ScrollStack, ScrollFloat, and ScrollReveal are JavaScript adaptations of the
 * React Bits source supplied for this project. CircuitBoard follows the public
 * API and visual behavior of Componentry's MIT-licensed Circuit Board:
 * https://componentry.dev/docs/components/circuit-board
 */

(function initializeTerraWalkExperience(global) {
    'use strict';

    const VIEW_IDS = {
        landing: 'landing-view',
        info: 'info-view',
        pipeline: 'pipeline-view'
    };

    const mountedRoots = new WeakMap();
    let components = null;
    let activeViewName = 'landing';

    function dependencySetIsReady() {
        return Boolean(
            global.React &&
            global.ReactDOM &&
            global.gsap &&
            global.ScrollTrigger
        );
    }

    function numberFromDataset(value, fallback) {
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : fallback;
    }

    function boolFromDataset(value, fallback) {
        if (value === undefined) return fallback;
        return value === 'true';
    }

    function currentViewIsVisible(viewName) {
        const viewId = VIEW_IDS[viewName];
        const view = viewId ? document.getElementById(viewId) : null;
        return Boolean(view && !view.classList.contains('hidden'));
    }

    function defineComponents() {
        if (components || !dependencySetIsReady()) return components;

        const React = global.React;
        const {
            createElement: h,
            useCallback,
            useEffect,
            useId,
            useLayoutEffect,
            useMemo,
            useRef
        } = React;
        const gsap = global.gsap;
        const ScrollTrigger = global.ScrollTrigger;

        gsap.registerPlugin(ScrollTrigger);

        function reducedMotionIsPreferred() {
            return Boolean(
                global.matchMedia &&
                global.matchMedia('(prefers-reduced-motion: reduce)').matches
            );
        }

        function ScrollFloat({
            children,
            scrollContainerRef,
            containerClassName = '',
            textClassName = '',
            animationDuration = 1,
            ease = 'back.inOut(2)',
            scrollStart = 'center bottom+=50%',
            scrollEnd = 'bottom bottom-=40%',
            stagger = 0.03,
            ownerView = 'landing'
        }) {
            const containerRef = useRef(null);
            const text = typeof children === 'string' ? children : '';
            const splitText = useMemo(
                () => text.split('').map((character, index) => h(
                    'span',
                    {
                        className: 'char',
                        key: `${character}-${index}`
                    },
                    character === ' ' ? '\u00A0' : character
                )),
                [text]
            );

            useEffect(() => {
                const element = containerRef.current;
                if (!element) return undefined;

                const characterElements = element.querySelectorAll('.char');
                const scroller = scrollContainerRef && scrollContainerRef.current
                    ? scrollContainerRef.current
                    : global;
                const animation = gsap.fromTo(
                    characterElements,
                    {
                        opacity: 0,
                        scaleX: 0.7,
                        scaleY: 2.3,
                        transformOrigin: '50% 0%',
                        willChange: 'opacity, transform',
                        yPercent: 120
                    },
                    {
                        duration: animationDuration,
                        ease,
                        opacity: 1,
                        scaleX: 1,
                        scaleY: 1,
                        stagger,
                        yPercent: 0,
                        scrollTrigger: {
                            end: scrollEnd,
                            scroller,
                            scrub: true,
                            start: scrollStart,
                            trigger: element
                        }
                    }
                );
                const motionQuery = global.matchMedia
                    ? global.matchMedia('(prefers-reduced-motion: reduce)')
                    : null;
                let ownerIsActive = currentViewIsVisible(ownerView);

                const syncAnimationState = () => {
                    const reduceMotion = Boolean(motionQuery?.matches);
                    const trigger = animation.scrollTrigger;

                    if (reduceMotion) {
                        if (trigger) trigger.disable(false);
                        animation.progress(1).pause();
                        gsap.set(characterElements, {
                            opacity: 1,
                            scaleX: 1,
                            scaleY: 1,
                            yPercent: 0
                        });
                    } else if (ownerIsActive) {
                        if (trigger) {
                            trigger.enable(false);
                            trigger.refresh();
                            trigger.update();
                        }
                    } else if (trigger) {
                        trigger.disable(false);
                    }
                };
                const handleViewChange = (event) => {
                    ownerIsActive = event.detail?.viewName === ownerView;
                    syncAnimationState();
                };
                const handleMotionChange = () => syncAnimationState();

                document.addEventListener('terrawalk:viewchange', handleViewChange);
                if (motionQuery) {
                    if (typeof motionQuery.addEventListener === 'function') {
                        motionQuery.addEventListener('change', handleMotionChange);
                    } else if (typeof motionQuery.addListener === 'function') {
                        motionQuery.addListener(handleMotionChange);
                    }
                }
                syncAnimationState();

                return () => {
                    document.removeEventListener(
                        'terrawalk:viewchange',
                        handleViewChange
                    );
                    if (motionQuery) {
                        if (typeof motionQuery.removeEventListener === 'function') {
                            motionQuery.removeEventListener(
                                'change',
                                handleMotionChange
                            );
                        } else if (typeof motionQuery.removeListener === 'function') {
                            motionQuery.removeListener(handleMotionChange);
                        }
                    }
                    if (animation.scrollTrigger) animation.scrollTrigger.kill();
                    animation.kill();
                };
            }, [
                animationDuration,
                ease,
                ownerView,
                scrollContainerRef,
                scrollEnd,
                scrollStart,
                stagger
            ]);

            return h(
                'h2',
                {
                    'aria-label': text,
                    className: `scroll-float ${containerClassName}`.trim(),
                    ref: containerRef
                },
                h(
                    'span',
                    {
                        'aria-hidden': 'true',
                        className: `scroll-float-text ${textClassName}`.trim()
                    },
                    splitText
                )
            );
        }

        function ScrollReveal({
            children,
            scrollContainerRef,
            enableBlur = true,
            baseOpacity = 0.1,
            baseRotation = 3,
            blurStrength = 4,
            containerClassName = '',
            textClassName = '',
            rotationEnd = 'bottom bottom',
            wordAnimationEnd = 'bottom bottom',
            ownerView = 'landing'
        }) {
            const containerRef = useRef(null);
            const text = typeof children === 'string' ? children : '';
            const splitText = useMemo(
                () => text.split(/(\s+)/).map((word, index) => {
                    if (/^\s+$/.test(word)) return word;
                    return h(
                        'span',
                        {
                            className: 'word',
                            key: `${word}-${index}`
                        },
                        word
                    );
                }),
                [text]
            );

            useEffect(() => {
                const element = containerRef.current;
                if (!element) return undefined;

                const wordElements = element.querySelectorAll('.word');
                const scroller = scrollContainerRef && scrollContainerRef.current
                    ? scrollContainerRef.current
                    : global;
                const animations = [];

                animations.push(gsap.fromTo(
                    element,
                    {
                        rotate: baseRotation,
                        transformOrigin: '0% 50%'
                    },
                    {
                        ease: 'none',
                        rotate: 0,
                        scrollTrigger: {
                            end: rotationEnd,
                            scroller,
                            scrub: true,
                            start: 'top bottom',
                            trigger: element
                        }
                    }
                ));

                animations.push(gsap.fromTo(
                    wordElements,
                    {
                        filter: enableBlur ? `blur(${blurStrength}px)` : 'none',
                        opacity: baseOpacity,
                        willChange: enableBlur ? 'opacity, filter' : 'opacity'
                    },
                    {
                        ease: 'none',
                        filter: enableBlur ? 'blur(0px)' : undefined,
                        opacity: 1,
                        stagger: 0.05,
                        scrollTrigger: {
                            end: wordAnimationEnd,
                            scroller,
                            scrub: true,
                            start: 'top bottom-=20%',
                            trigger: element
                        }
                    }
                ));
                const motionQuery = global.matchMedia
                    ? global.matchMedia('(prefers-reduced-motion: reduce)')
                    : null;
                let ownerIsActive = currentViewIsVisible(ownerView);

                const syncAnimationState = () => {
                    const reduceMotion = Boolean(motionQuery?.matches);

                    if (reduceMotion) {
                        animations.forEach((animation) => {
                            if (animation.scrollTrigger) {
                                animation.scrollTrigger.disable(false);
                            }
                            animation.progress(1).pause();
                        });
                        gsap.set(element, { rotate: 0 });
                        gsap.set(wordElements, {
                            filter: 'blur(0px)',
                            opacity: 1
                        });
                    } else if (ownerIsActive) {
                        animations.forEach((animation) => {
                            if (animation.scrollTrigger) {
                                animation.scrollTrigger.enable(false);
                                animation.scrollTrigger.refresh();
                                animation.scrollTrigger.update();
                            }
                        });
                    } else {
                        animations.forEach((animation) => {
                            if (animation.scrollTrigger) {
                                animation.scrollTrigger.disable(false);
                            }
                        });
                    }
                };
                const handleViewChange = (event) => {
                    ownerIsActive = event.detail?.viewName === ownerView;
                    syncAnimationState();
                };
                const handleMotionChange = () => syncAnimationState();

                document.addEventListener('terrawalk:viewchange', handleViewChange);
                if (motionQuery) {
                    if (typeof motionQuery.addEventListener === 'function') {
                        motionQuery.addEventListener('change', handleMotionChange);
                    } else if (typeof motionQuery.addListener === 'function') {
                        motionQuery.addListener(handleMotionChange);
                    }
                }
                syncAnimationState();

                return () => {
                    document.removeEventListener(
                        'terrawalk:viewchange',
                        handleViewChange
                    );
                    if (motionQuery) {
                        if (typeof motionQuery.removeEventListener === 'function') {
                            motionQuery.removeEventListener(
                                'change',
                                handleMotionChange
                            );
                        } else if (typeof motionQuery.removeListener === 'function') {
                            motionQuery.removeListener(handleMotionChange);
                        }
                    }
                    animations.forEach((animation) => {
                        if (animation.scrollTrigger) animation.scrollTrigger.kill();
                        animation.kill();
                    });
                };
            }, [
                baseOpacity,
                baseRotation,
                blurStrength,
                enableBlur,
                ownerView,
                rotationEnd,
                scrollContainerRef,
                wordAnimationEnd
            ]);

            return h(
                'div',
                {
                    className: `scroll-reveal ${containerClassName}`.trim(),
                    ref: containerRef
                },
                h(
                    'p',
                    {
                        className: `scroll-reveal-text ${textClassName}`.trim()
                    },
                    splitText
                )
            );
        }

        function ScrollStackItem({ children, itemClassName = '' }) {
            return h(
                'article',
                {
                    className: `scroll-stack-card ${itemClassName}`.trim()
                },
                children
            );
        }

        function ScrollStack({
            children,
            className = '',
            itemDistance = 100,
            itemScale = 0.03,
            itemStackDistance = 30,
            stackPosition = '20%',
            scaleEndPosition = '10%',
            baseScale = 0.85,
            scaleDuration = 0.5,
            rotationAmount = 0,
            blurAmount = 0,
            useWindowScroll = false,
            onStackComplete,
            ownerView = 'landing',
            ariaLabel = 'Scrollable card stack'
        }) {
            const scrollerRef = useRef(null);
            const stackCompletedRef = useRef(false);
            const animationFrameRef = useRef(null);
            const lenisRef = useRef(null);
            const cardsRef = useRef([]);
            const lastTransformsRef = useRef(new Map());
            const isUpdatingRef = useRef(false);
            const activeRef = useRef(currentViewIsVisible(ownerView));
            const reducedMotion = reducedMotionIsPreferred();
            const motionReducedRef = useRef(reducedMotion);

            const calculateProgress = useCallback((scrollTop, start, end) => {
                if (scrollTop < start) return 0;
                if (scrollTop > end) return 1;
                if (end === start) return 1;
                return (scrollTop - start) / (end - start);
            }, []);

            const parsePercentage = useCallback((value, containerHeight) => {
                if (typeof value === 'string' && value.includes('%')) {
                    return (parseFloat(value) / 100) * containerHeight;
                }
                return parseFloat(value);
            }, []);

            const getScrollData = useCallback(() => {
                if (useWindowScroll) {
                    return {
                        containerHeight: global.innerHeight,
                        scrollContainer: document.documentElement,
                        scrollTop: global.scrollY
                    };
                }
                const scroller = scrollerRef.current;
                return {
                    containerHeight: scroller ? scroller.clientHeight : 0,
                    scrollContainer: scroller,
                    scrollTop: scroller ? scroller.scrollTop : 0
                };
            }, [useWindowScroll]);

            const getElementOffset = useCallback((element) => {
                if (useWindowScroll) {
                    let offsetTop = 0;
                    let currentElement = element;
                    while (currentElement) {
                        offsetTop += currentElement.offsetTop || 0;
                        currentElement = currentElement.offsetParent;
                    }
                    return offsetTop;
                }
                return element.offsetTop;
            }, [useWindowScroll]);

            const updateCardTransforms = useCallback(() => {
                if (
                    motionReducedRef.current ||
                    !activeRef.current ||
                    !cardsRef.current.length ||
                    isUpdatingRef.current
                ) {
                    return;
                }

                isUpdatingRef.current = true;
                const {
                    scrollTop,
                    containerHeight
                } = getScrollData();
                const stackPositionPx = parsePercentage(stackPosition, containerHeight);
                const scaleEndPositionPx = parsePercentage(scaleEndPosition, containerHeight);
                const scroller = scrollerRef.current;
                const endElement = scroller
                    ? scroller.querySelector('.scroll-stack-end')
                    : null;
                const endElementTop = endElement ? getElementOffset(endElement) : 0;

                let topCardIndex = 0;
                cardsRef.current.forEach((card, index) => {
                    const cardTop = getElementOffset(card);
                    const triggerStart = cardTop - stackPositionPx - itemStackDistance * index;
                    if (scrollTop >= triggerStart) topCardIndex = index;
                });

                cardsRef.current.forEach((card, index) => {
                    if (!card) return;

                    const cardTop = getElementOffset(card);
                    const triggerStart = cardTop - stackPositionPx - itemStackDistance * index;
                    const triggerEnd = cardTop - scaleEndPositionPx;
                    const pinStart = triggerStart;
                    const pinEnd = endElementTop - containerHeight / 2;
                    const scaleProgress = calculateProgress(
                        scrollTop,
                        triggerStart,
                        triggerEnd
                    );
                    const targetScale = baseScale + index * itemScale;
                    const scale = 1 - scaleProgress * (1 - targetScale);
                    const rotation = rotationAmount
                        ? index * rotationAmount * scaleProgress
                        : 0;
                    const blur = blurAmount && index < topCardIndex
                        ? Math.max(0, (topCardIndex - index) * blurAmount)
                        : 0;

                    let translateY = 0;
                    const isPinned = scrollTop >= pinStart && scrollTop <= pinEnd;
                    if (isPinned) {
                        translateY = scrollTop - cardTop + stackPositionPx +
                            itemStackDistance * index;
                    } else if (scrollTop > pinEnd) {
                        translateY = pinEnd - cardTop + stackPositionPx +
                            itemStackDistance * index;
                    }

                    const nextTransform = {
                        blur: Math.round(blur * 100) / 100,
                        rotation: Math.round(rotation * 100) / 100,
                        scale: Math.round(scale * 1000) / 1000,
                        translateY: Math.round(translateY * 100) / 100
                    };
                    const lastTransform = lastTransformsRef.current.get(index);
                    const changed = !lastTransform ||
                        Math.abs(lastTransform.translateY - nextTransform.translateY) > 0.1 ||
                        Math.abs(lastTransform.scale - nextTransform.scale) > 0.001 ||
                        Math.abs(lastTransform.rotation - nextTransform.rotation) > 0.1 ||
                        Math.abs(lastTransform.blur - nextTransform.blur) > 0.1;

                    if (changed) {
                        card.style.transform =
                            `translate3d(0, ${nextTransform.translateY}px, 0) ` +
                            `scale(${nextTransform.scale}) rotate(${nextTransform.rotation}deg)`;
                        card.style.filter = nextTransform.blur > 0
                            ? `blur(${nextTransform.blur}px)`
                            : '';
                        lastTransformsRef.current.set(index, nextTransform);
                    }

                    if (index === cardsRef.current.length - 1) {
                        const isInView = scrollTop >= pinStart && scrollTop <= pinEnd;
                        if (isInView && !stackCompletedRef.current) {
                            stackCompletedRef.current = true;
                            if (typeof onStackComplete === 'function') onStackComplete();
                        } else if (!isInView && stackCompletedRef.current) {
                            stackCompletedRef.current = false;
                        }
                    }
                });

                isUpdatingRef.current = false;
            }, [
                baseScale,
                blurAmount,
                calculateProgress,
                getElementOffset,
                getScrollData,
                itemScale,
                itemStackDistance,
                onStackComplete,
                parsePercentage,
                rotationAmount,
                scaleEndPosition,
                stackPosition
            ]);

            useLayoutEffect(() => {
                const scroller = scrollerRef.current;
                if (!scroller) return undefined;

                const cards = Array.from(
                    scroller.querySelectorAll('.scroll-stack-card')
                );
                cardsRef.current = cards;
                cards.forEach((card, index) => {
                    if (index < cards.length - 1) {
                        card.style.marginBottom = `${itemDistance}px`;
                    }
                    card.style.backfaceVisibility = 'hidden';
                    card.style.perspective = '1000px';
                    card.style.transformOrigin = 'top center';
                    card.style.transition = `box-shadow ${scaleDuration}s ease`;
                    card.style.webkitPerspective = '1000px';
                    if (reducedMotion) {
                        card.style.filter = 'none';
                        card.style.transform = 'none';
                        card.style.webkitTransform = 'none';
                        card.style.willChange = 'auto';
                    } else {
                        card.style.transform = 'translateZ(0)';
                        card.style.webkitTransform = 'translateZ(0)';
                        card.style.willChange = 'transform, filter';
                    }
                });

                const handleScroll = () => updateCardTransforms();
                let nativeScrollFallback = false;
                let runLenisFrame = null;
                const motionQuery = global.matchMedia
                    ? global.matchMedia('(prefers-reduced-motion: reduce)')
                    : null;

                const stopLenisFrame = () => {
                    if (animationFrameRef.current !== null) {
                        global.cancelAnimationFrame(animationFrameRef.current);
                        animationFrameRef.current = null;
                    }
                };

                const startLenisFrame = () => {
                    if (
                        motionReducedRef.current ||
                        !activeRef.current ||
                        !lenisRef.current ||
                        animationFrameRef.current !== null
                    ) {
                        return;
                    }

                    runLenisFrame = (time) => {
                        if (!activeRef.current || !lenisRef.current) {
                            animationFrameRef.current = null;
                            return;
                        }
                        lenisRef.current.raf(time);
                        animationFrameRef.current =
                            global.requestAnimationFrame(runLenisFrame);
                    };
                    animationFrameRef.current =
                        global.requestAnimationFrame(runLenisFrame);
                };

                if (!reducedMotion && global.Lenis && !useWindowScroll) {
                    const lenisOptions = {
                        content: scroller.querySelector('.scroll-stack-inner'),
                        duration: 1.2,
                        easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
                        infinite: false,
                        lerp: 0.1,
                        smoothWheel: true,
                        syncTouch: true,
                        syncTouchLerp: 0.075,
                        touchMultiplier: 2,
                        wheelMultiplier: 1,
                        wrapper: scroller
                    };
                    const lenis = new global.Lenis(lenisOptions);
                    lenis.on('scroll', handleScroll);
                    lenisRef.current = lenis;
                    if (activeRef.current) {
                        startLenisFrame();
                    } else {
                        lenis.stop();
                    }
                }
                nativeScrollFallback = true;
                const nativeScrollTarget = useWindowScroll ? global : scroller;
                nativeScrollTarget.addEventListener('scroll', handleScroll, {
                    passive: true
                });

                const handleViewChange = (event) => {
                    activeRef.current = event.detail?.viewName === ownerView;
                    if (lenisRef.current) {
                        if (activeRef.current && !motionReducedRef.current) {
                            lenisRef.current.start();
                            startLenisFrame();
                        } else {
                            lenisRef.current.stop();
                            stopLenisFrame();
                        }
                    }
                    if (activeRef.current && !motionReducedRef.current) {
                        global.requestAnimationFrame(() => {
                            updateCardTransforms();
                        });
                    }
                };
                const handleMotionChange = (event) => {
                    motionReducedRef.current = event.matches;

                    if (event.matches) {
                        if (lenisRef.current) {
                            lenisRef.current.destroy();
                            lenisRef.current = null;
                        }
                        stopLenisFrame();
                        cards.forEach((card) => {
                            card.style.filter = 'none';
                            card.style.transform = 'none';
                            card.style.webkitTransform = 'none';
                            card.style.willChange = 'auto';
                        });
                        lastTransformsRef.current.clear();
                    } else if (activeRef.current) {
                        cards.forEach((card) => {
                            card.style.willChange = 'transform, filter';
                        });
                        if (lenisRef.current) {
                            lenisRef.current.start();
                            startLenisFrame();
                        }
                        global.requestAnimationFrame(updateCardTransforms);
                    }
                };
                document.addEventListener('terrawalk:viewchange', handleViewChange);
                if (motionQuery) {
                    if (typeof motionQuery.addEventListener === 'function') {
                        motionQuery.addEventListener('change', handleMotionChange);
                    } else if (typeof motionQuery.addListener === 'function') {
                        motionQuery.addListener(handleMotionChange);
                    }
                }

                updateCardTransforms();

                return () => {
                    document.removeEventListener(
                        'terrawalk:viewchange',
                        handleViewChange
                    );
                    if (motionQuery) {
                        if (typeof motionQuery.removeEventListener === 'function') {
                            motionQuery.removeEventListener(
                                'change',
                                handleMotionChange
                            );
                        } else if (typeof motionQuery.removeListener === 'function') {
                            motionQuery.removeListener(handleMotionChange);
                        }
                    }
                    stopLenisFrame();
                    if (lenisRef.current) {
                        lenisRef.current.destroy();
                        lenisRef.current = null;
                    }
                    if (nativeScrollFallback) {
                        nativeScrollTarget.removeEventListener(
                            'scroll',
                            handleScroll
                        );
                    }
                    cardsRef.current = [];
                    lastTransformsRef.current.clear();
                    stackCompletedRef.current = false;
                    isUpdatingRef.current = false;
                };
            }, [
                itemDistance,
                ownerView,
                reducedMotion,
                scaleDuration,
                updateCardTransforms,
                useWindowScroll
            ]);

            return h(
                'div',
                {
                    'aria-label': ariaLabel,
                    className: [
                        'scroll-stack-scroller',
                        useWindowScroll ? 'scroll-stack-scroller--window' : '',
                        className
                    ].filter(Boolean).join(' '),
                    ref: scrollerRef,
                    role: 'region',
                    tabIndex: useWindowScroll ? undefined : 0
                },
                h(
                    'div',
                    {
                        className: 'scroll-stack-inner'
                    },
                    children,
                    h('div', {
                        'aria-hidden': 'true',
                        className: 'scroll-stack-end'
                    })
                )
            );
        }

        function makeOrthogonalPath(startNode, endNode) {
            const midpointX = startNode.x + (endNode.x - startNode.x) / 2;
            return [
                `M ${startNode.x} ${startNode.y}`,
                `L ${midpointX} ${startNode.y}`,
                `L ${midpointX} ${endNode.y}`,
                `L ${endNode.x} ${endNode.y}`
            ].join(' ');
        }

        function CircuitBoard({
            nodes,
            connections,
            width = 600,
            height = 400,
            showGrid = true,
            pulseSpeed = 2,
            traceWidth = 2,
            className = ''
        }) {
            const componentId = useId().replace(/:/g, '');
            const boardDescriptionId = `${componentId}-description`;
            const nodeMap = useMemo(
                () => new Map(nodes.map((node) => [node.id, node])),
                [nodes]
            );
            const reduceMotion = reducedMotionIsPreferred();
            const renderedConnections = connections.map((connection, index) => {
                const startNode = nodeMap.get(connection.from);
                const endNode = nodeMap.get(connection.to);
                if (!startNode || !endNode) return null;

                const pathData = makeOrthogonalPath(startNode, endNode);
                const pathId = `${componentId}-trace-${index}`;
                const traceKind = connection.kind || 'request';
                const traceClass = [
                    'circuit-board-trace',
                    traceKind === 'fallback' ? 'circuit-board-trace--fallback' : '',
                    traceKind === 'response' ? 'circuit-board-trace--response' : ''
                ].filter(Boolean).join(' ');
                const pulseClass = [
                    'circuit-board-pulse',
                    traceKind === 'fallback' ? 'circuit-board-pulse--fallback' : '',
                    traceKind === 'response' ? 'circuit-board-pulse--response' : ''
                ].filter(Boolean).join(' ');
                const elements = [
                    h('path', {
                        className: 'circuit-board-trace-shadow',
                        d: pathData,
                        key: `${pathId}-shadow`,
                        strokeWidth: traceWidth + 4
                    }),
                    h('path', {
                        className: traceClass,
                        d: pathData,
                        id: pathId,
                        key: `${pathId}-trace`,
                        strokeWidth: traceWidth
                    })
                ];

                if (connection.animated && !reduceMotion) {
                    elements.push(h(
                        'circle',
                        {
                            className: pulseClass,
                            key: `${pathId}-pulse`,
                            r: 3.25
                        },
                        h('animateMotion', {
                            begin: `${index * 0.18}s`,
                            dur: `${pulseSpeed}s`,
                            path: pathData,
                            repeatCount: 'indefinite'
                        })
                    ));
                }

                if (connection.bidirectional && !reduceMotion) {
                    elements.push(h(
                        'circle',
                        {
                            className: `${pulseClass} circuit-board-pulse--response`,
                            key: `${pathId}-return-pulse`,
                            r: 2.8
                        },
                        h('animateMotion', {
                            begin: `${0.45 + index * 0.18}s`,
                            dur: `${pulseSpeed * 1.15}s`,
                            keyPoints: '1;0',
                            keyTimes: '0;1',
                            path: pathData,
                            repeatCount: 'indefinite'
                        })
                    ));
                }

                return h(
                    'g',
                    {
                        key: pathId
                    },
                    elements
                );
            });

            const renderedNodes = nodes.map((node) => {
                const size = node.size === 'lg' ? 66 : node.size === 'sm' ? 46 : 56;
                const half = size / 2;
                const status = node.status || 'inactive';
                return h(
                    'g',
                    {
                        className: `circuit-node circuit-node--${status}`,
                        key: node.id,
                        transform: `translate(${node.x} ${node.y})`
                    },
                    h('circle', {
                        className: 'circuit-node-halo',
                        r: half + 9
                    }),
                    h('rect', {
                        className: 'circuit-node-body',
                        height: size,
                        rx: 13,
                        width: size,
                        x: -half,
                        y: -half
                    }),
                    h(
                        'text',
                        {
                            className: 'circuit-node-icon',
                            dy: '0.35em',
                            x: 0,
                            y: 0
                        },
                        node.icon || node.id.slice(0, 2).toUpperCase()
                    ),
                    h(
                        'text',
                        {
                            className: 'circuit-node-label',
                            x: 0,
                            y: half + 21
                        },
                        node.label || node.id
                    ),
                    h('circle', {
                        className: 'circuit-node-status',
                        cx: half - 3,
                        cy: -half + 3,
                        r: 4
                    })
                );
            });

            return h(
                'div',
                {
                    className: `pipeline-board-frame ${className}`.trim()
                },
                h(
                    'p',
                    {
                        className: 'experience-visually-hidden',
                        id: boardDescriptionId
                    },
                    'Eight-stage command flow. Operator and terrain data enter the browser payload, ' +
                    'then Flask validates the request. A configured Groq provider can produce the ' +
                    'strict-schema response; otherwise the deterministic fallback uses the same ' +
                    'contract. Both branches converge at backend sanitization, versioned JSON, ' +
                    'browser correlation checks, and finally the Three.js telemetry and controller.'
                ),
                h(
                    'div',
                    {
                        className: 'pipeline-board-scroll'
                    },
                    h(
                        'svg',
                        {
                            'aria-label': 'TerraWalk command and kinematic response pipeline',
                            'aria-describedby': boardDescriptionId,
                            className: 'circuit-board',
                            preserveAspectRatio: 'xMidYMid meet',
                            role: 'img',
                            viewBox: `0 0 ${width} ${height}`
                        },
                        h(
                            'defs',
                            null,
                            h(
                                'pattern',
                                {
                                    height: 20,
                                    id: `${componentId}-dot-grid`,
                                    patternUnits: 'userSpaceOnUse',
                                    width: 20
                                },
                                h('circle', {
                                    className: 'circuit-board-grid-dot',
                                    cx: 1,
                                    cy: 1,
                                    r: 1
                                })
                            )
                        ),
                        showGrid
                            ? h('rect', {
                                fill: `url(#${componentId}-dot-grid)`,
                                height: '100%',
                                width: '100%'
                            })
                            : null,
                        renderedConnections,
                        renderedNodes
                    )
                ),
                h(
                    'div',
                    {
                        'aria-label': 'Circuit board legend',
                        className: 'circuit-board-legend'
                    },
                    h('span', null, h('i'), 'Primary request path'),
                    h(
                        'span',
                        null,
                        h('i', { className: 'legend-fallback' }),
                        'Deterministic fallback'
                    ),
                    h(
                        'span',
                        null,
                        h('i', { className: 'legend-response' }),
                        'Validated response'
                    )
                )
            );
        }

        components = {
            CircuitBoard,
            ScrollFloat,
            ScrollReveal,
            ScrollStack,
            ScrollStackItem,
            h
        };
        return components;
    }

    function mountReactRoot(rootElement, reactElement) {
        if (!rootElement || mountedRoots.has(rootElement)) return;

        try {
            const reactRoot = global.ReactDOM.createRoot(rootElement);
            reactRoot.render(reactElement);
            rootElement.classList.add('react-bits-enhanced');
            mountedRoots.set(rootElement, reactRoot);
        } catch (error) {
            rootElement.classList.remove('react-bits-enhanced');
            console.warn('[Experience]: React island retained its static fallback.', error);
        }
    }

    function mountScrollFloat(rootElement, viewName) {
        const { ScrollFloat, h } = components;
        const text = rootElement.dataset.text ||
            rootElement.textContent.trim().replace(/\s+/g, ' ');
        mountReactRoot(rootElement, h(
            ScrollFloat,
            {
                animationDuration: numberFromDataset(
                    rootElement.dataset.animationDuration,
                    1
                ),
                containerClassName: rootElement.dataset.containerClass || '',
                ease: rootElement.dataset.ease || 'back.inOut(2)',
                key: `${viewName}-scroll-float`,
                ownerView: viewName,
                scrollEnd: rootElement.dataset.scrollEnd || 'bottom bottom-=25%',
                scrollStart: rootElement.dataset.scrollStart || 'top bottom-=10%',
                stagger: numberFromDataset(rootElement.dataset.stagger, 0.025),
                textClassName: rootElement.dataset.textClass || ''
            },
            text
        ));
    }

    function mountScrollReveal(rootElement, viewName) {
        const { ScrollReveal, h } = components;
        const text = rootElement.dataset.text ||
            rootElement.textContent.trim().replace(/\s+/g, ' ');
        mountReactRoot(rootElement, h(
            ScrollReveal,
            {
                baseOpacity: numberFromDataset(rootElement.dataset.baseOpacity, 0.12),
                baseRotation: numberFromDataset(rootElement.dataset.baseRotation, 2),
                blurStrength: numberFromDataset(rootElement.dataset.blurStrength, 5),
                containerClassName: rootElement.dataset.containerClass || '',
                enableBlur: boolFromDataset(rootElement.dataset.enableBlur, true),
                key: `${viewName}-scroll-reveal`,
                ownerView: viewName,
                rotationEnd: rootElement.dataset.rotationEnd || 'bottom center',
                textClassName: rootElement.dataset.textClass || '',
                wordAnimationEnd:
                    rootElement.dataset.wordAnimationEnd || 'bottom center'
            },
            text
        ));
    }

    function mountScrollStack(rootElement, viewName) {
        const {
            ScrollStack,
            ScrollStackItem,
            h
        } = components;
        const useWindowScroll = boolFromDataset(
            rootElement.dataset.useWindowScroll,
            true
        );
        const cardMarkup = Array.from(
            rootElement.querySelectorAll('[data-scroll-stack-card]')
        ).map((card, index) => ({
            html: card.innerHTML,
            itemClassName: card.dataset.itemClass || '',
            key: card.dataset.cardKey || `${viewName}-stack-${index}`
        }));

        if (!cardMarkup.length) return;

        rootElement.classList.toggle(
            'experience-stack-root--window',
            useWindowScroll
        );

        const items = cardMarkup.map((card) => h(
            ScrollStackItem,
            {
                itemClassName: card.itemClassName,
                key: card.key
            },
            h('div', {
                className: 'stack-card-content',
                dangerouslySetInnerHTML: { __html: card.html }
            })
        ));

        mountReactRoot(rootElement, h(
            ScrollStack,
            {
                ariaLabel: rootElement.dataset.ariaLabel || 'Scrollable story cards',
                baseScale: numberFromDataset(rootElement.dataset.baseScale, 0.88),
                blurAmount: numberFromDataset(rootElement.dataset.blurAmount, 0.45),
                itemDistance: numberFromDataset(rootElement.dataset.itemDistance, 88),
                itemScale: numberFromDataset(rootElement.dataset.itemScale, 0.025),
                itemStackDistance: numberFromDataset(
                    rootElement.dataset.itemStackDistance,
                    26
                ),
                ownerView: viewName,
                rotationAmount: numberFromDataset(
                    rootElement.dataset.rotationAmount,
                    0.35
                ),
                scaleDuration: numberFromDataset(
                    rootElement.dataset.scaleDuration,
                    0.45
                ),
                scaleEndPosition: rootElement.dataset.scaleEndPosition || '12%',
                stackPosition: rootElement.dataset.stackPosition || '14%',
                useWindowScroll
            },
            items
        ));
    }

    function mountCircuitBoard(rootElement) {
        const { CircuitBoard, h } = components;
        const nodes = [
            {
                icon: 'OP',
                id: 'operator',
                label: 'Operator + terrain',
                size: 'md',
                status: 'active',
                x: 74,
                y: 245
            },
            {
                icon: 'UI',
                id: 'browser',
                label: 'Browser payload',
                size: 'md',
                status: 'active',
                x: 225,
                y: 245
            },
            {
                icon: 'API',
                id: 'flask',
                label: 'Flask validation',
                size: 'lg',
                status: 'active',
                x: 398,
                y: 245
            },
            {
                icon: 'AI',
                id: 'groq',
                label: 'Groq provider branch',
                size: 'md',
                status: 'standby',
                x: 590,
                y: 110
            },
            {
                icon: 'FB',
                id: 'fallback',
                label: 'Safe local control',
                size: 'md',
                status: 'standby',
                x: 590,
                y: 375
            },
            {
                icon: '✓',
                id: 'sanitizer',
                label: 'Sanitize + override',
                size: 'lg',
                status: 'active',
                x: 780,
                y: 245
            },
            {
                icon: '3.2',
                id: 'response',
                label: 'Versioned JSON',
                size: 'md',
                status: 'active',
                x: 940,
                y: 245
            },
            {
                icon: '3D',
                id: 'three',
                label: 'Three.js + telemetry',
                size: 'md',
                status: 'active',
                x: 1072,
                y: 245
            }
        ];
        const connections = [
            { animated: true, from: 'operator', to: 'browser' },
            {
                animated: true,
                bidirectional: true,
                from: 'browser',
                to: 'flask'
            },
            { animated: true, from: 'flask', to: 'groq' },
            {
                animated: true,
                from: 'flask',
                kind: 'fallback',
                to: 'fallback'
            },
            { animated: true, from: 'groq', to: 'sanitizer' },
            {
                animated: true,
                from: 'fallback',
                kind: 'fallback',
                to: 'sanitizer'
            },
            {
                animated: true,
                from: 'sanitizer',
                kind: 'response',
                to: 'response'
            },
            {
                animated: true,
                from: 'response',
                kind: 'response',
                to: 'three'
            }
        ];

        mountReactRoot(rootElement, h(CircuitBoard, {
            connections,
            height: 500,
            nodes,
            pulseSpeed: 2.2,
            showGrid: true,
            traceWidth: 2.2,
            width: 1140
        }));
    }

    function mountView(viewName) {
        const viewId = VIEW_IDS[viewName];
        const view = viewId ? document.getElementById(viewId) : null;
        if (!view || !components) return;

        view.querySelectorAll('[data-react-bits="scroll-float"]').forEach(
            (root) => mountScrollFloat(root, viewName)
        );
        view.querySelectorAll('[data-react-bits="scroll-reveal"]').forEach(
            (root) => mountScrollReveal(root, viewName)
        );
        view.querySelectorAll('[data-react-bits="scroll-stack"]').forEach(
            (root) => mountScrollStack(root, viewName)
        );
        view.querySelectorAll('[data-componentry="circuit-board"]').forEach(
            (root) => mountCircuitBoard(root)
        );
    }

    function activateView(viewName) {
        activeViewName = viewName;

        if (components && VIEW_IDS[viewName]) mountView(viewName);
        document.dispatchEvent(new CustomEvent('terrawalk:viewchange', {
            detail: { viewName }
        }));

        global.requestAnimationFrame(() => {
            if (VIEW_IDS[viewName] && global.ScrollTrigger) {
                global.ScrollTrigger.refresh();
            }
        });
    }

    function initialize() {
        if (!dependencySetIsReady()) {
            console.warn(
                '[Experience]: Animation dependencies unavailable; static content remains active.'
            );
            return;
        }

        defineComponents();
        const dashboard = document.getElementById('dashboard-view');
        const dashboardIsVisible = Boolean(
            dashboard && !dashboard.classList.contains('hidden')
        );
        const visibleView = Object.keys(VIEW_IDS).find(currentViewIsVisible) ||
            (dashboardIsVisible ? 'dashboard' : 'landing');
        activateView(visibleView);
    }

    global.activateTerraWalkExperience = activateView;
    global.refreshTerraWalkExperience = function refreshTerraWalkExperience() {
        activateView(activeViewName);
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize, { once: true });
    } else {
        initialize();
    }
}(window));
