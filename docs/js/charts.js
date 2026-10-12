// Renders every <div data-fig="name"> from figs/name.{light|dark}.json and redraws when
// the colour scheme changes. Specs are written by src/elevenwinters/figures/build.py.
(function () {
  var media = window.matchMedia("(prefers-color-scheme: dark)");
  function mode() {
    var t = document.documentElement.getAttribute("data-theme");
    if (t === "dark" || t === "light") return t;
    return media.matches ? "dark" : "light";
  }
  var base = document.body.getAttribute("data-root") || "";
  function draw(el) {
    var name = el.getAttribute("data-fig");
    return fetch(base + "figs/" + name + "." + mode() + ".json")
      .then(function (r) { return r.json(); })
      .then(function (spec) { return vegaEmbed(el, spec, { actions: false, renderer: "svg" }); })
      .catch(function () { el.textContent = "This chart could not be loaded. Reload the page to try again."; });
  }
  function drawAll() { document.querySelectorAll("[data-fig]").forEach(draw); }
  window.ewMode = mode;
  if (media.addEventListener) media.addEventListener("change", drawAll);
  new MutationObserver(drawAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  drawAll();
})();
