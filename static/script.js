const themeToggle = document.getElementById("themeToggle");
const themeIcon = document.getElementById("themeIcon");
const themeText = document.getElementById("themeText");
const searchForm = document.getElementById("searchForm");
const bookTitle = document.getElementById("bookTitle");
const loadingState = document.getElementById("loadingState");
const recommendButton = document.getElementById("recommendButton");

function applyTheme(theme) {
    const isDark = theme === "dark";
    document.body.classList.toggle("dark", isDark);
    themeIcon.textContent = isDark ? "☀️" : "🌙";
    themeText.textContent = isDark ? "Light" : "Dark";
}

const savedTheme = localStorage.getItem("bookRecommenderTheme");
applyTheme(savedTheme === "dark" ? "dark" : "light");

themeToggle.addEventListener("click", function () {
    const nextTheme = document.body.classList.contains("dark") ? "light" : "dark";
    localStorage.setItem("bookRecommenderTheme", nextTheme);
    applyTheme(nextTheme);
});

searchForm.addEventListener("submit", function () {
    if (!bookTitle.value.trim()) {
        return;
    }
    loadingState.hidden = false;
    recommendButton.disabled = true;
    recommendButton.textContent = "Searching...";
});

document.querySelectorAll(".demo-chip").forEach(function (chip) {
    chip.addEventListener("click", function () {
        bookTitle.value = chip.dataset.query;
        searchForm.requestSubmit();
    });
});
