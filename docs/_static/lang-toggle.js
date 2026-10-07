// Korean / English switch.
//
// The site is built once, with the Korean pages mirroring the English ones under
// /ko/. The button swaps the "ko/" segment in and out of the current path, and a
// class on <body> hides the other language's section of the sidebar.
(function () {
  function rootRelative() {
    // DOCUMENTATION_OPTIONS.URL_ROOT is "../" per directory level, so it tells us
    // where the site root is relative to this page.
    var root = (window.DOCUMENTATION_OPTIONS && DOCUMENTATION_OPTIONS.URL_ROOT) || "./";
    var here = window.location.pathname;
    var depth = (root.match(/\.\.\//g) || []).length;
    var parts = here.split("/");
    parts.pop();                                   // drop the file name
    var base = parts.slice(0, parts.length - depth).join("/") + "/";
    return {base: base, page: here.slice(base.length)};
  }

  function build() {
    var loc = rootRelative();
    var isKorean = loc.page.indexOf("ko/") === 0;
    var other = isKorean ? loc.page.slice(3) : "ko/" + loc.page;

    document.body.classList.add(isKorean ? "lang-ko" : "lang-en");

    // Tag each sidebar caption with the language it introduces, so the CSS can
    // show only one of the two trees. Captions written in Hangul are the Korean ones.
    document.querySelectorAll(".wy-menu-vertical p.caption").forEach(function (c) {
      c.setAttribute("data-lang", /[\uac00-\ud7a3]/.test(c.textContent) ? "ko" : "en");
    });

    var box = document.querySelector(".wy-side-nav-search");
    if (!box) return;
    var wrap = document.createElement("div");
    wrap.className = "lang-switch";
    wrap.innerHTML =
      '<a href="' + loc.base + (isKorean ? loc.page.slice(3) : loc.page) + '"' +
      (isKorean ? "" : ' class="active"') + ">English</a>" +
      '<a href="' + loc.base + (isKorean ? loc.page : "ko/" + loc.page) + '"' +
      (isKorean ? ' class="active"' : "") + ">한국어</a>";
    box.appendChild(wrap);

    // A Korean page that has no English twin (or the other way round) would 404;
    // fall back to that language's home page when the file is not there.
    wrap.querySelectorAll("a").forEach(function (a) {
      a.addEventListener("click", function (e) {
        var url = a.getAttribute("href");
        e.preventDefault();
        fetch(url, {method: "HEAD"}).then(function (r) {
          window.location.href = r.ok ? url : loc.base + (a.textContent === "한국어" ? "ko/index.html" : "index.html");
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
