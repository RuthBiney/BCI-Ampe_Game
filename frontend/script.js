const startBtn = document.getElementById("start-btn");
const predictBtn = document.getElementById("predict-btn");
const gameArea = document.getElementById("game-area");
const movementDisplay = document.getElementById("movement");
const scoreDisplay = document.getElementById("score");
const eegUpload = document.getElementById("eeg-upload");

let score = 0;

startBtn.addEventListener("click", () => {
  gameArea.style.display = "block";
  startBtn.style.display = "none";
});

predictBtn.addEventListener("click", async () => {
  const files = eegUpload.files;
  if (files.length === 0) {
    alert("Please upload EEG data files first!");
    return;
  }

  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();
    const prediction = result.predictions[0];

    if (prediction) {
      const predictedMovement = prediction.movement;
      movementDisplay.textContent = predictedMovement;

      // Increment score based on movement type
      if (
        predictedMovement === "left hand" ||
        predictedMovement === "right hand"
      ) {
        score += 15;
      } else {
        score += 5;
      }

      scoreDisplay.textContent = score;
    } else {
      movementDisplay.textContent = "Unknown";
    }
  } catch (error) {
    console.error("Prediction Error:", error);
    alert("Failed to predict movement!");
  }
});
