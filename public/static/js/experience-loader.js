/*
 * Loads the optional non-simulation experience after the core document is ready.
 *
 * Keeping these dependencies out of parser-blocking and deferred head scripts lets
 * the existing Three.js console initialize immediately. If a CDN is unavailable,
 * the semantic HTML fallbacks remain visible and fully usable.
 */

(function loadTerraWalkExperience(global, document) {
    'use strict';

    const loaderScript = document.currentScript;
    const experienceSource = loaderScript?.dataset.experienceSrc;
    const pendingScripts = new Map();

    function loadScript(source) {
        if (!source) return Promise.reject(new Error('Missing script source.'));
        if (pendingScripts.has(source)) return pendingScripts.get(source);

        const request = new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.async = true;
            script.src = source;
            script.onload = () => resolve(script);
            script.onerror = () => reject(new Error(`Unable to load ${source}`));
            document.head.appendChild(script);
        });

        pendingScripts.set(source, request);
        return request;
    }

    async function hydrateExperience() {
        try {
            const reactReady = global.React
                ? Promise.resolve()
                : loadScript(
                    'https://unpkg.com/react@18.3.1/umd/react.production.min.js'
                );
            const gsapReady = global.gsap
                ? Promise.resolve()
                : loadScript(
                    'https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js'
                );

            await Promise.all([
                reactReady.then(() => (
                    global.ReactDOM
                        ? Promise.resolve()
                        : loadScript(
                            'https://unpkg.com/react-dom@18.3.1/umd/react-dom.production.min.js'
                        )
                )),
                global.Lenis
                    ? Promise.resolve()
                    : loadScript(
                        'https://cdn.jsdelivr.net/npm/lenis@1.1.20/dist/lenis.min.js'
                    ),
                gsapReady.then(() => (
                    global.ScrollTrigger
                        ? Promise.resolve()
                        : loadScript(
                            'https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/ScrollTrigger.min.js'
                        )
                ))
            ]);

            await loadScript(experienceSource);
        } catch (error) {
            console.warn(
                '[Experience]: Optional enhancements could not load; static content remains active.',
                error
            );
        }
    }

    function scheduleHydration() {
        if (typeof global.requestIdleCallback === 'function') {
            global.requestIdleCallback(hydrateExperience, { timeout: 900 });
        } else {
            global.setTimeout(hydrateExperience, 0);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', scheduleHydration, {
            once: true
        });
    } else {
        scheduleHydration();
    }
}(window, document));
