document.getElementById("predict-form").addEventListener("submit", async function (e) {
  e.preventDefault();
  const location = document.getElementById("location").value;

  const response = await fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ location: location })
  });

  const data = await response.json();
  document.getElementById("result").textContent = data.forecast;
});
