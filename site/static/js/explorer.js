// City explorer: pick a monitoring site, see its observed and weather-normalised PM2.5.
// Data: data/sites.json (index) and data/sites/<site>.json, written by the site builder.
(function () {
  var input = document.getElementById("site-input");
  var list = document.getElementById("site-list");
  var status = document.getElementById("site-status");
  var chartEl = document.getElementById("explorer-chart");
  var seasonEl = document.getElementById("explorer-season");
  var sites = [];
  var current = null;

  function theme() {
    return fetch("figs/_theme." + window.ewMode() + ".json").then(function (r) { return r.json(); });
  }

  function label(s) { return s.name + " · " + s.city + (s.network === "us_mission" ? " (US mission)" : ""); }

  function rows(d) {
    var out = [];
    for (var i = 0; i < d.date.length; i++) {
      out.push({ date: d.date[i], kind: "Observed", pm25: d.observed_30d[i] });
      out.push({ date: d.date[i], kind: "Weather removed", pm25: d.normalised_30d[i] });
    }
    return out;
  }

  function render(d) {
    theme().then(function (t) {
      var p = t.palette;
      var series = {
        $schema: "https://vega.github.io/schema/vega-lite/v6.json",
        config: t.config, width: 640, height: 260,
        data: { values: rows(d) },
        mark: { type: "line", interpolate: "monotone" },
        encoding: {
          x: { field: "date", type: "temporal", title: null, axis: { format: "%Y", tickCount: "year" } },
          y: { field: "pm25", type: "quantitative", title: "PM2.5, 30-day mean, µg/m³" },
          color: { field: "kind", type: "nominal", title: null,
                   scale: { domain: ["Observed", "Weather removed"], range: [p.faint, p.ink] } },
          strokeWidth: { field: "kind", type: "nominal", legend: null,
                         scale: { domain: ["Observed", "Weather removed"], range: [1.2, 2.2] } },
          tooltip: [{ field: "date", type: "temporal", format: "%d %b %Y" }, { field: "kind" },
                    { field: "pm25", type: "quantitative", format: ".0f" }]
        }
      };
      var season = {
        $schema: "https://vega.github.io/schema/vega-lite/v6.json",
        config: t.config, width: 640, height: 180,
        data: { values: d.monthly },
        mark: { type: "line", point: { filled: true, size: 36 } },
        encoding: {
          x: { field: "month", type: "ordinal", title: null, sort: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"] },
          y: { field: "pm25", type: "quantitative", title: "Average by month, µg/m³" },
          color: { field: "kind", type: "nominal", title: null,
                   scale: { domain: ["Observed", "Weather removed"], range: [p.faint, p.ink] } }
        }
      };
      vegaEmbed(chartEl, series, { actions: false, renderer: "svg" });
      vegaEmbed(seasonEl, season, { actions: false, renderer: "svg" });
    });
  }

  function load(site) {
    var s = sites.find(function (x) { return x.site === site; });
    if (!s) { status.textContent = "No site matches that name. Pick one from the list."; return; }
    current = site;
    status.textContent = s.days + " days with data, " + s.first + " to " + s.last + ". Model skill (cross-validated R² on log PM2.5): " + s.r2 + ".";
    fetch("data/sites/" + site + ".json").then(function (r) { return r.json(); }).then(render);
    try { localStorage.setItem("ew-site", site); } catch (e) {}
  }

  fetch("data/sites.json").then(function (r) { return r.json(); }).then(function (idx) {
    sites = idx;
    idx.forEach(function (s) {
      var o = document.createElement("option");
      o.value = label(s);
      list.appendChild(o);
    });
    var start = "us_delhi";
    try { start = localStorage.getItem("ew-site") || start; } catch (e) {}
    var first = idx.find(function (s) { return s.site === start; }) || idx[0];
    input.value = label(first);
    load(first.site);
  });

  input.addEventListener("change", function () {
    var s = sites.find(function (x) { return label(x) === input.value; });
    if (s) load(s.site); else status.textContent = "No site matches that name. Pick one from the list.";
  });
  var media = window.matchMedia("(prefers-color-scheme: dark)");
  if (media.addEventListener) media.addEventListener("change", function () { if (current) load(current); });
})();
