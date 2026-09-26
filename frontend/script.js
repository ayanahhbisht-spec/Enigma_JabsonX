```javascript
/* =========================================================
   FEDRISK
   Federated Financial Intelligence
   script.js
   ========================================================= */

document.addEventListener("DOMContentLoaded", () => {

    /* =====================================================
       NAVBAR — SCROLL EFFECT
       ===================================================== */

    const navbar = document.querySelector(".navbar");

    if (navbar) {
        const updateNavbar = () => {
            if (window.scrollY > 40) {
                navbar.style.background = "rgba(10, 10, 10, 0.94)";
                navbar.style.borderBottomColor =
                    "rgba(59, 130, 246, 0.12)";
            } else {
                navbar.style.background = "rgba(10, 10, 10, 0.82)";
                navbar.style.borderBottomColor =
                    "rgba(148, 163, 184, 0.10)";
            }
        };

        window.addEventListener("scroll", updateNavbar, {
            passive: true
        });

        updateNavbar();
    }


    /* =====================================================
       SMOOTH SCROLL
       ===================================================== */

    document.querySelectorAll('a[href^="#"]').forEach(link => {

        link.addEventListener("click", event => {

            const targetId = link.getAttribute("href");

            if (!targetId || targetId === "#") {
                return;
            }

            const target = document.querySelector(targetId);

            if (!target) {
                return;
            }

            event.preventDefault();

            const navbarHeight =
                navbar ? navbar.offsetHeight : 0;

            const targetPosition =
                target.getBoundingClientRect().top +
                window.scrollY -
                navbarHeight;

            window.scrollTo({
                top: targetPosition,
                behavior: "smooth"
            });

        });

    });


    /* =====================================================
       SCROLL REVEAL
       ===================================================== */

    const revealElements = document.querySelectorAll(
        ".feature-card, " +
        ".network-intro, " +
        ".institution-node, " +
        ".shared-model, " +
        ".risk-content, " +
        ".dashboard-wrapper, " +
        ".privacy-header, " +
        ".privacy-visual, " +
        ".cta-content"
    );

    revealElements.forEach(element => {

        element.style.opacity = "0";
        element.style.transform =
            "translateY(25px)";

        element.style.transition =
            "opacity 0.7s ease, transform 0.7s ease";

    });


    const revealObserver = new IntersectionObserver(
        entries => {

            entries.forEach(entry => {

                if (!entry.isIntersecting) {
                    return;
                }

                entry.target.style.opacity = "1";
                entry.target.style.transform =
                    "translateY(0)";

                revealObserver.unobserve(entry.target);

            });

        },
        {
            threshold: 0.12,
            rootMargin: "0px 0px -50px 0px"
        }
    );


    revealElements.forEach(element => {
        revealObserver.observe(element);
    });


    /* =====================================================
       STAGGER FEATURE CARDS
       ===================================================== */

    const featureCards =
        document.querySelectorAll(".feature-card");

    featureCards.forEach((card, index) => {

        card.style.transitionDelay =
            `${index * 100}ms`;

    });


    /* =====================================================
       NETWORK NODE HOVER
       ===================================================== */

    const institutionNodes =
        document.querySelectorAll(".institution-node");

    institutionNodes.forEach(node => {

        node.addEventListener("mouseenter", () => {

            institutionNodes.forEach(other => {

                if (other !== node) {
                    other.style.opacity = "0.45";
                }

            });

        });

        node.addEventListener("mouseleave", () => {

            institutionNodes.forEach(other => {
                other.style.opacity = "1";
            });

        });

    });


    /* =====================================================
       FEDERATED NETWORK — DYNAMIC DATA PULSES
       ===================================================== */

    const flowLines =
        document.querySelectorAll(".flow-line");

    flowLines.forEach(line => {

        const pulse = line.querySelector(".flow-pulse");

        if (!pulse) {
            return;
        }

        setInterval(() => {

            pulse.style.animation = "none";

            // Force browser repaint
            void pulse.offsetWidth;

            pulse.style.animation =
                "flowPulse 2.4s linear";

        }, 4000);

    });


    /* =====================================================
       DASHBOARD — LIVE METRIC SIMULATION
       ===================================================== */

    const roundElement =
        document.querySelector(".round-info strong");

    const dataMovement =
        document.querySelector(".data-movement strong");

    const signalValue =
        document.querySelector(".signal-value");

    const metricLine =
        document.querySelector(".metric-line span");

    let trainingRound = 128;

    let dataValue = 94.7;

    let signal = 82;


    function updateDashboard() {

        trainingRound++;

        if (trainingRound > 999) {
            trainingRound = 100;
        }

        dataValue +=
            (Math.random() * 0.4) - 0.2;

        dataValue =
            Math.max(
                92,
                Math.min(98, dataValue)
            );

        signal +=
            Math.floor(
                (Math.random() * 5) - 2
            );

        signal =
            Math.max(
                70,
                Math.min(98, signal)
            );


        if (roundElement) {
            roundElement.textContent =
                trainingRound
                    .toString()
                    .padStart(3, "0");
        }

        if (dataMovement) {
            dataMovement.textContent =
                `${dataValue.toFixed(1)}%`;
        }

        if (signalValue) {
            signalValue.textContent =
                `${signal}%`;
        }

        if (metricLine) {
            metricLine.style.width =
                `${signal}%`;
        }

    }


    /*
       Update slowly so it feels like
       a live intelligence dashboard
       instead of a random counter.
    */

    setInterval(updateDashboard, 5000);


    /* =====================================================
       CHART — SUBTLE LIVE MOVEMENT
       ===================================================== */

    const chartPolyline =
        document.querySelector(".chart-svg polyline");

    if (chartPolyline) {

        let chartPoints = [
            66,
            61,
            65,
            55,
            58,
            49,
            43,
            46,
            37,
            31
        ];


        function updateChart() {

            chartPoints.shift();

            const previous =
                chartPoints[chartPoints.length - 1];

            let next =
                previous +
                ((Math.random() * 12) - 6);

            next =
                Math.max(
                    15,
                    Math.min(75, next)
                );

            chartPoints.push(next);


            const width = 100;

            const step =
                width / (chartPoints.length - 1);


            const points =
                chartPoints
                    .map((value, index) => {

                        const x =
                            index * step;

                        return `${x},${value}`;

                    })
                    .join(" ");


            chartPolyline.setAttribute(
                "points",
                points
            );

        }


        setInterval(updateChart, 3000);

    }


    /* =====================================================
       CHART POINT PULSE
       ===================================================== */

    const chartPoints =
        document.querySelectorAll(".chart-point");

    chartPoints.forEach((point, index) => {

        point.animate(
            [
                {
                    opacity: 0.35,
                    transform: "scale(0.8)"
                },
                {
                    opacity: 1,
                    transform: "scale(1.2)"
                },
                {
                    opacity: 0.35,
                    transform: "scale(0.8)"
                }
            ],
            {
                duration: 1800,
                delay: index * 300,
                iterations: Infinity,
                easing: "ease-in-out"
            }
        );

    });


    /* =====================================================
       MINI NETWORK — DATA ACTIVITY
       ===================================================== */

    const miniNodes =
        document.querySelectorAll(
            ".mini-institution .institution-dot"
        );

    miniNodes.forEach((node, index) => {

        setInterval(() => {

            node.animate(
                [
                    {
                        opacity: 0.25,
                        transform: "scale(0.7)"
                    },
                    {
                        opacity: 1,
                        transform: "scale(1.3)"
                    },
                    {
                        opacity: 0.25,
                        transform: "scale(0.7)"
                    }
                ],
                {
                    duration: 1000,
                    easing: "ease-in-out"
                }
            );

        }, 1800 + index * 500);

    });


    /* =====================================================
       PRIVACY FLOW
       ===================================================== */

    const privacyFlow =
        document.querySelector(".privacy-flow");

    if (privacyFlow) {

        const privacyDots =
            privacyFlow.querySelectorAll("span");

        privacyDots.forEach((dot, index) => {

            setInterval(() => {

                dot.animate(
                    [
                        {
                            opacity: 0.25,
                            transform: "scale(0.7)"
                        },
                        {
                            opacity: 1,
                            transform: "scale(1.3)"
                        },
                        {
                            opacity: 0.25,
                            transform: "scale(0.7)"
                        }
                    ],
                    {
                        duration: 1400,
                        delay: index * 200,
                        easing: "ease-in-out"
                    }
                );

            }, 2500);

        });

    }


    /* =====================================================
       CTA NETWORK — MOVING CENTER PULSE
       ===================================================== */

    const ctaCenter =
        document.querySelector(".cta-node-5");

    if (ctaCenter) {

        setInterval(() => {

            ctaCenter.animate(
                [
                    {
                        transform: "scale(0.8)",
                        opacity: 0.4
                    },
                    {
                        transform: "scale(1.5)",
                        opacity: 1
                    },
                    {
                        transform: "scale(0.8)",
                        opacity: 0.4
                    }
                ],
                {
                    duration: 1800,
                    easing: "ease-in-out"
                }
            );

        }, 2200);

    }


    /* =====================================================
       CTA / PRIMARY BUTTON INTERACTION
       ===================================================== */

    const demoButtons =
        document.querySelectorAll(
            ".primary-button, .nav-button"
        );

    demoButtons.forEach(button => {

        button.addEventListener("click", event => {

            const href =
                button.getAttribute("href");

            /*
              If the button has a real anchor target,
              let smooth-scroll handle it.
            */

            if (href && href.startsWith("#")) {
                return;
            }

            /*
              Otherwise show a small status effect.
            */

            const originalText =
                button.innerHTML;

            button.innerHTML =
                "CONNECTING <span>→</span>";

            button.style.pointerEvents = "none";

            setTimeout(() => {

                button.innerHTML =
                    originalText;

                button.style.pointerEvents =
                    "auto";

            }, 1400);

        });

    });


    /* =====================================================
       CARD TILT
       ===================================================== */

    const tiltCards =
        document.querySelectorAll(
            ".feature-card, .dashboard"
        );


    tiltCards.forEach(card => {

        card.addEventListener("mousemove", event => {

            /*
              Disable strong tilt on mobile.
            */

            if (window.innerWidth < 800) {
                return;
            }

            const rect =
                card.getBoundingClientRect();

            const x =
                event.clientX - rect.left;

            const y =
                event.clientY - rect.top;

            const centerX =
                rect.width / 2;

            const centerY =
                rect.height / 2;

            const rotateX =
                ((y - centerY) / centerY) * -2;

            const rotateY =
                ((x - centerX) / centerX) * 2;


            if (card.classList.contains("dashboard")) {

                card.style.transform =
                    `perspective(1400px)
                     rotateX(${rotateX}deg)
                     rotateY(${rotateY - 3}deg)`;

            } else {

                card.style.transform =
                    `perspective(1000px)
                     rotateX(${rotateX}deg)
                     rotateY(${rotateY}deg)
                     translateY(-3px)`;

            }

        });


        card.addEventListener("mouseleave", () => {

            if (
                card.classList.contains("dashboard")
            ) {

                card.style.transform =
                    "perspective(1400px) rotateY(-3deg)";

            } else {

                card.style.transform =
                    "perspective(1000px) rotateX(0deg) rotateY(0deg)";

            }

        });

    });


    /* =====================================================
       PARALLAX NETWORK
       ===================================================== */

    const heroNetwork =
        document.querySelector(".hero-network");

    if (heroNetwork) {

        window.addEventListener(
            "scroll",
            () => {

                if (window.innerWidth < 800) {
                    return;
                }

                const scroll =
                    window.scrollY;

                heroNetwork.style.transform =
                    `translateY(${scroll * 0.08}px)`;

            },
            {
                passive: true
            }
        );

    }


    /* =====================================================
       ACTIVE NAVIGATION
       ===================================================== */

    const sections =
        document.querySelectorAll(
            "section[id]"
        );

    const navLinks =
        document.querySelectorAll(
            ".nav-links a"
        );


    const sectionObserver =
        new IntersectionObserver(
            entries => {

                entries.forEach(entry => {

                    if (!entry.isIntersecting) {
                        return;
                    }

                    const id =
                        entry.target.getAttribute("id");

                    navLinks.forEach(link => {

                        link.classList.remove(
                            "active"
                        );

                        if (
                            link.getAttribute("href") ===
                            `#${id}`
                        ) {

                            link.classList.add(
                                "active"
                            );

                        }

                    });

                });

            },
            {
                threshold: 0.35
            }
        );


    sections.forEach(section => {
        sectionObserver.observe(section);
    });


    /* =====================================================
       ACTIVE NAV STYLE
       ===================================================== */

    const activeStyle =
        document.createElement("style");

    activeStyle.textContent = `
        .nav-links a.active {
            color: #FFFFFF;
        }

        .nav-links a.active::after {
            width: 100%;
        }
    `;

    document.head.appendChild(activeStyle);


    /* =====================================================
       SYSTEM STATUS CLOCK
       ===================================================== */

    const footerMeta =
        document.querySelector(".footer-meta");

    if (footerMeta) {

        const clock =
            document.createElement("span");

        clock.className =
            "live-clock";

        footerMeta.appendChild(clock);


        function updateClock() {

            const now =
                new Date();

            const hours =
                now.getHours()
                    .toString()
                    .padStart(2, "0");

            const minutes =
                now.getMinutes()
                    .toString()
                    .padStart(2, "0");

            const seconds =
                now.getSeconds()
                    .toString()
                    .padStart(2, "0");

            clock.textContent =
                `SYSTEM TIME // ${hours}:${minutes}:${seconds}`;

        }


        updateClock();

        setInterval(
            updateClock,
            1000
        );

    }


    /* =====================================================
       KEYBOARD ACCESSIBILITY
       ===================================================== */

    document.addEventListener(
        "keydown",
        event => {

            /*
              Press "/" to jump to the main
              content / network section.
            */

            if (
                event.key === "/" &&
                document.activeElement.tagName !== "INPUT" &&
                document.activeElement.tagName !== "TEXTAREA"
            ) {

                const target =
                    document.querySelector(
                        "#network"
                    );

                if (target) {

                    event.preventDefault();

                    target.scrollIntoView({
                        behavior: "smooth"
                    });

                }

            }

        }
    );


    /* =====================================================
       PAGE READY
       ===================================================== */

    document.body.classList.add(
        "page-loaded"
    );

});
```
