const form = document.getElementById("uploadForm");
const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const processedImage = document.getElementById("processedImage");
const predictedText = document.getElementById("predictedText");

const previewCard = document.getElementById("previewCard");
const processedColumn = document.getElementById("processedColumn");

function clearAllImages() {
  previewImage.src = "";
  processedImage.src = "";
  predictedText.textContent = "";
  previewCard.classList.add("d-none");
  processedColumn.classList.add("d-none");
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];

  if (!file) {
    clearAllImages();
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    previewImage.src = reader.result;
    previewCard.classList.remove("d-none");

    processedImage.src = "";
    predictedText.textContent = "";
    processedColumn.classList.add("d-none");
  };
  reader.readAsDataURL(file);
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const file = imageInput.files[0];
  if (!file) {
    alert("Please select an image before submitting.");
    return;
  }

  processedColumn.classList.add("d-none");
  predictedText.textContent = "";
  processedImage.src = "";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }

    const data = await response.json();

    if (data.processed_image) {
      processedImage.src = "data:image/png;base64," + data.processed_image;
      predictedText.textContent = data.full_prediction;

      processedColumn.classList.remove("d-none");
      previewCard.classList.remove("d-none");
    } else {
      alert("Failed to receive processed image.");
    }
  } catch (error) {
    alert("Error: " + error.message);
  }
});
