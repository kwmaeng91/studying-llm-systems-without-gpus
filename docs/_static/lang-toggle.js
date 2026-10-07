// Korean / English switch.
//
// The site is built once: the English pages sit at the root and the Korean ones
// mirror them under /ko/. The button swaps that segment in and out of the current
// path, and a class on <body> hides the other language's section of the sidebar.
(function () {
  // Where the site root is. The theme's home link is relative to the current
  // page, so resolving it against the current URL gives the root — on the home
  // page itself that link is "#", which resolves to the page's own URL. This
  // avoids depending on DOCUMENTATION_OPTIONS.URL_ROOT, which Sphinx removed.
  function siteLocation() {
    var home = document.querySelector(".wy-side-nav-search a.icon-home") ||
               document.querySelector("a.icon-home");
    var href = home ? home.getAttribute("href") : "index.html";
    if (!href || href === "#") href = window.location.href;
    var base = new URL(href, window.location.href).pathname.replace(/[^/]*$/, "");
    var here = window.location.pathname;
    var page = here.indexOf(base) === 0 ? here.slice(base.length) : here.replace(/^\//, "");
    return {base: base, page: page};
  }

  function build() {
    var loc = siteLocation();
    var isKorean = loc.page === "ko" || loc.page.indexOf("ko/") === 0;
    var pages = {
      en: loc.base + (isKorean ? loc.page.replace(/^ko\/?/, "") : loc.page),
      ko: loc.base + (isKorean ? loc.page : "ko/" + loc.page)
    };

    document.body.classList.add(isKorean ? "lang-ko" : "lang-en");

    // Tag each sidebar caption with the language it introduces, so the CSS can
    // show only one of the two trees. Captions in Hangul are the Korean ones.
    document.querySelectorAll(".wy-menu-vertical p.caption").forEach(function (c) {
      c.setAttribute("data-lang", /[\uac00-\ud7a3]/.test(c.textContent) ? "ko" : "en");
    });

    var box = document.querySelector(".wy-side-nav-search");
    if (!box) return;
    var wrap = document.createElement("div");
    wrap.className = "lang-switch";
    wrap.innerHTML =
      '<a href="' + pages.en + '" data-lang="en"' + (isKorean ? "" : ' class="active"') + ">English</a>" +
      '<a href="' + pages.ko + '" data-lang="ko"' + (isKorean ? ' class="active"' : "") + ">한국어</a>";
    box.appendChild(wrap);

    wrap.querySelectorAll("a").forEach(function (a) {
      a.addEventListener("click", function (e) {
        e.preventDefault();
        if (a.classList.contains("active")) return;      // already in this language
        var url = a.getAttribute("href");
        var home = loc.base + (a.dataset.lang === "ko" ? "ko/index.html" : "index.html");
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
