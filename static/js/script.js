document.getElementById('imageInput').addEventListener('change', function (event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            const queryImage = document.getElementById('queryImage');
            queryImage.src = e.target.result;
        }
        reader.readAsDataURL(file);
    }
});

window.addEventListener('load', function () {
    const queryContainer = document.getElementById('query-container');
    const optionContainer = document.getElementById('option-container');
    const mainContainer = document.getElementById('container');

    function equalizeHeights() {
        const queryHeight = queryContainer.offsetHeight;
        const optionHeight = optionContainer.offsetHeight;
        const mainHeight = mainContainer.offsetHeight;
        const maxHeight = Math.max(queryHeight, optionHeight, mainHeight);

        queryContainer.style.height = maxHeight + 'px';
        optionContainer.style.height = maxHeight + 'px';
        mainContainer.style.height = maxHeight + 'px';
    }

    equalizeHeights();
    window.addEventListener('resize', equalizeHeights);
});

document.getElementById('url-query').addEventListener('input', function (event) {
    const queryImage = document.getElementById('queryImage');
    queryImage.src = document.getElementById('url-query').value;
});

function showDescribe(img) {
    const describeDiv = img.nextElementSibling.nextElementSibling;
    describeDiv.classList.add('show');
}

function closeDescribe(button) {
    const describeDiv = button.parentElement.parentElement;
    describeDiv.classList.remove('show');
}

