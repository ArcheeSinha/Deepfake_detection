document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const spinner = document.getElementById('spinner');
    spinner.style.display = 'block';

    const fileInput = document.querySelector('input[type="file"]');
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    const xhr = new XMLHttpRequest();

    xhr.upload.addEventListener('progress', function(e) {
        if (e.lengthComputable) {
            const percentComplete = (e.loaded / e.total) * 100;
            document.getElementById('progress-bar').style.width = percentComplete + '%';
        }
    }, false);

    xhr.addEventListener('load', function() {
        spinner.style.display = 'none';
        showModal();
    }, false);

    xhr.open('POST', 'Upload', true); 
    xhr.send(formData);
});

function showModal() {
    const modal = document.getElementById("uploadModal");
    const span = document.querySelector(".close");
    const okBtn = document.getElementById("modal-ok-btn");

    modal.style.display = "block";

    span.onclick = function() {
        modal.style.display = "none";
    }

    okBtn.onclick = function() {
        modal.style.display = "none";
    }

    window.onclick = function(event) {
        if (event.target == modal) {
            modal.style.display = "none";
        }
    }
}
