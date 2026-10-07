// Korean / English switch.
//
// The site is built once: the English pages sit at the root and the Korean ones
// mirror them under /ko/. The button swaps that segment in and out of the current
// path, and a class on <body> hides the other language's section of the sidebar.
(function () {
  // Where the site root is, taken from this script's own URL. Sphinx links
  // _static/ with the right number of "../" on every page, and the browser has
  // already resolved it — if it had not, this script would not be running. That
  // makes it reliable at any depth and under any hosting prefix (Read the Docs
  // serves the site under /<lang>/<version>/, for instance), unlike
  // DOCUMENTATION_OPTIONS.URL_ROOT, which Sphinx 9 no longer sets.
  var self = document.currentScript ||
             document.querySelector('script[src*="lang-toggle.js"]');
  var BASE = new URL(self.src, window.location.href).pathname
               .replace(/_static\/lang-toggle\.js.*$/, "");

  function build() {
    var here = window.location.pathname;
    var page = here.indexOf(BASE) === 0 ? here.slice(BASE.length) : "";
    var isKorean = page === "ko" || page.indexOf("ko/") === 0;
    var target = {
      en: BASE + (isKorean ? page.replace(/^ko\/?/, "") : page),
      ko: BASE + (isKorean ? page : "ko/" + page)
    };

    document.body.classList.add(isKorean ? "lang-ko" : "lang-en");

    // Tag each sidebar caption with the language it introduces, so the CSS can
    // show only one of the two trees. Captions in Hangul are the Korean ones.
    document.querySelectorAll(".wy-menu-vertical p.caption").forEach(function (c) {
      c.setAttribute("data-lang", /[가-힣]/.test(c.textContent) ? "ko" : "en");
    });

    var box = document.querySelector(".wy-side-nav-search");
    if (!box) return;
    var wrap = document.createElement("div");
    wrap.className = "lang-switch";
    wrap.innerHTML =
      '<a href="' + target.en + '" data-lang="en"' + (isKorean ? "" : ' class="active"') + ">English</a>" +
      '<a href="' + target.ko + '" data-lang="ko"' + (isKorean ? ' class="active"' : "") + ">한국어</a>";
    box.appendChild(wrap);

    wrap.querySelectorAll("a").forEach(function (a) {
      a.addEventListener("click", function (e) {
        e.preventDefault();
        if (a.classList.contains("active")) return;      // already in this language
        var url = a.getAttribute("href");
        var home = BASE + (a.dataset.lang === "ko" ? "ko/" : "");
        // A page with no counterpart (the untranslated draft lecture, say) would
        // 404, so fall back to that language's home page.
        fetch(url, {method: "HEAD"}).then(function (r) {
          window.location.href = r.status === 404 ? home : url;
        }).catch(function () { window.location.href = url; });
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", build);
  } else {
    build();
  }
})();
