const startBtn = document.getElementById("startBtn");
const video = document.getElementById("video");
const text = document.getElementById("text");

startBtn.onclick = () => {
  video.src = "http://127.0.0.1:8000/video";
  setInterval(fetchPrediction, 500);
};

function fetchPrediction() {
  fetch("http://127.0.0.1:8000/prediction")
    .then(res => res.json())
    .then(data => {
      text.innerText = data.prediction;
    });
}