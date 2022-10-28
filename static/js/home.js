const gaugeElement1 = document.querySelector(".gauge1");
const gaugeElement2 = document.querySelector(".gauge2");
const gaugeElement3 = document.querySelector(".gauge3");



function setGaugeValue(gauge, value) {
  if (value < 0 || value > 1) {
    return;
  }

  gauge.querySelector(".gauge__fill").style.transform = `rotate(${
    value / 2
  }turn)`;

  gauge.querySelector(".gauge__cover").textContent = `${Math.round(
    value * 100
  )}%`;
}

setGaugeValue(gaugeElement1, 0.87);
setGaugeValue(gaugeElement2, 0.92);
setGaugeValue(gaugeElement3, 0.94);