document.addEventListener("DOMContentLoaded", function() {
  console.log("JavaScript code executed.");
  const dropArea = document.querySelector(".drop_box");
  const button = dropArea.querySelector("#choose-file");
  // const dragText = dropArea.querySelector("header");
  const input = dropArea.querySelector("input");
  const fileInfo = dropArea.querySelector(".file-info");
  // const loadingIndicator = document.getElementById("loading-indicator");
  // const outputSection = document.getElementById("output-section");
  // const extractedText = document.getElementById("extracted-text");
  const zoomIframe = document.getElementById("zoom-iframe");
  button.addEventListener("click", () => {
    input.click();
  });

  input.addEventListener("change", function(e) {
    const fileName = e.target.files[0].name;
    const fileData = `
      <div class="file-info">
        <h4>Selected File:</h4>
        <p>${fileName}</p>
      </div>`;

    if (fileInfo) {
      fileInfo.remove();
    }

    dropArea.insertAdjacentHTML("beforeend", fileData);
  // Display selected file
  if (e.target.files && e.target.files[0]) {
    const fileURL = URL.createObjectURL(e.target.files[0]);
    zoomIframe.src = fileURL;
    imageContainer.style.display = "block";
  }
});

  const imageContainer = document.querySelector(".image-container");
  const zoomImage = document.querySelector(".zoom-image");

  const zooming = new Zooming({
    bgColor: "rgba(0, 0, 0, 0.8)",
    zIndex: 9999,
  });

  zooming.listen(zoomImage);

  imageContainer.addEventListener("click", function() {
    zooming.open(zoomImage);
  });

  
  
  
});
