// Tema compartido de ECharts para toda la app (estilo editorial claro).
// Uso: SentinelCharts.render("id-del-div", opcionesEcharts)
(function () {
  const css = getComputedStyle(document.documentElement);
  const v = (name) => css.getPropertyValue(name).trim();

  const c = {
    ink: v("--ink"), ink2: v("--ink-2"), ink3: v("--ink-3"), rule: v("--rule"),
    pos: v("--pos"), neg: v("--neg"), info: v("--info"), highlight: v("--highlight"), card: v("--card"),
  };
  const mono = "JetBrains Mono, ui-monospace, monospace";
  const sans = "Inter, system-ui, sans-serif";

  const axis = {
    axisLine: { lineStyle: { color: c.ink } },
    axisTick: { show: false },
    axisLabel: { color: c.ink2, fontFamily: mono, fontSize: 11 },
    splitLine: { lineStyle: { color: c.rule } },
    nameTextStyle: { color: c.ink3, fontFamily: mono, fontSize: 10 },
  };

  echarts.registerTheme("sentinel", {
    color: [c.pos, c.neg, c.info, c.ink3, c.highlight],
    backgroundColor: "transparent",
    textStyle: { fontFamily: sans, color: c.ink },
    legend: { textStyle: { color: c.ink2, fontFamily: sans, fontSize: 12 }, icon: "roundRect", itemWidth: 12, itemHeight: 12 },
    tooltip: {
      backgroundColor: c.card, borderColor: c.ink, borderWidth: 1,
      textStyle: { color: c.ink, fontFamily: mono, fontSize: 12 },
      extraCssText: "border-radius:4px;box-shadow:0 4px 18px rgba(0,0,0,.08)",
    },
    categoryAxis: { ...axis, splitLine: { show: false }, axisLabel: { ...axis.axisLabel, fontFamily: sans, fontSize: 12 } },
    valueAxis: { ...axis, axisLine: { show: false } },
  });

  const charts = [];
  window.addEventListener("resize", () => charts.forEach((ch) => ch.resize()));

  window.SentinelCharts = {
    c,
    render(id, option) {
      const el = document.getElementById(id);
      if (!el) return null;
      const ch = echarts.init(el, "sentinel", { renderer: "svg" });
      ch.setOption({ animationDuration: 900, animationEasing: "cubicOut", ...option });
      charts.push(ch);
      return ch;
    },
  };
})();
