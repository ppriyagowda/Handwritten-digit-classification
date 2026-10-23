// ==========================================
// 1. GET HTML ELEMENTS
// ==========================================

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const resultElement = document.getElementById("result");
const confidenceElement = document.getElementById("confidence");
const liveHistoryElement = document.getElementById("liveHistory");
const tableElement = document.getElementById("table");


// ==========================================
// 2. CANVAS DRAWING
// ==========================================

ctx.lineWidth = 18;
ctx.lineCap = "round";
ctx.lineJoin = "round";
ctx.strokeStyle = "white";

let drawing = false;
let lastX = 0;
let lastY = 0;

function getPosition(event) {
    const rect = canvas.getBoundingClientRect();

    let clientX;
    let clientY;

    if (event.touches && event.touches.length > 0) {
        clientX = event.touches[0].clientX;
        clientY = event.touches[0].clientY;
    } else {
        clientX = event.clientX;
        clientY = event.clientY;
    }

    return {
        x: (clientX - rect.left) * canvas.width / rect.width,
        y: (clientY - rect.top) * canvas.height / rect.height
    };
}

function startDrawing(event) {
    event.preventDefault();

    drawing = true;

    const position = getPosition(event);

    lastX = position.x;
    lastY = position.y;

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);

    // Draw a dot even if the user only taps the canvas.
    ctx.lineTo(lastX + 0.1, lastY + 0.1);
    ctx.stroke();
}

function draw(event) {
    if (!drawing) return;

    event.preventDefault();

    const position = getPosition(event);

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(position.x, position.y);
    ctx.stroke();

    lastX = position.x;
    lastY = position.y;
}

function stopDrawing() {
    drawing = false;
}

canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseleave", stopDrawing);

canvas.addEventListener("touchstart", startDrawing, { passive: false });
canvas.addEventListener("touchmove", draw, { passive: false });
canvas.addEventListener("touchend", stopDrawing);
canvas.addEventListener("touchcancel", stopDrawing);


// ==========================================
// 3. CLEAR CANVAS
// ==========================================

function clearCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    resultElement.textContent = "Predicted Digit: -";
    confidenceElement.textContent = "Confidence: -";

    const top3Element = document.getElementById("top3Results");

    if (top3Element) {
        top3Element.innerHTML = "";
    }
}


// ==========================================
// 4. CHART INITIALIZATION
// ==========================================

let history = [];

let chart = null;

if (typeof Chart !== "undefined") {
    chart = new Chart(document.getElementById("chart"), {
        type: "bar",

        data: {
            labels: [],
            datasets: [{
                label: "Predicted Digit",
                data: [],
                backgroundColor: "#22c55e"
            }]
        },

        options: {
            responsive: true,

            scales: {
                y: {
                    beginAtZero: true,
                    min: 0,
                    max: 9,
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });
}


// ==========================================
// 5. DISPLAY TOP THREE PREDICTIONS
// ==========================================

function displayTop3(top3) {
    let top3Element = document.getElementById("top3Results");

    if (!top3Element) {
        top3Element = document.createElement("div");
        top3Element.id = "top3Results";

        confidenceElement.insertAdjacentElement(
            "afterend",
            top3Element
        );
    }

    top3Element.innerHTML = "<h4>Top 3 Predictions</h4>";

    if (!Array.isArray(top3)) {
        top3Element.innerHTML += "<p>Top 3 predictions unavailable.</p>";
        return;
    }

    top3.forEach(function (item, index) {
        const row = document.createElement("p");

        row.textContent =
            (index + 1) + ". Digit " + item.digit +
            " — " + Number(item.confidence).toFixed(2) + "%";

        top3Element.appendChild(row);
    });
}


// ==========================================
// 6. VOICE PREDICTION
// ==========================================

function speakPrediction(digit) {
    if (!("speechSynthesis" in window)) return;

    window.speechSynthesis.cancel();

    const message = new SpeechSynthesisUtterance(
        "Predicted digit is " + digit
    );

    window.speechSynthesis.speak(message);
}


// ==========================================
// 7. PREDICT DIGIT
// ==========================================

async function predict() {
    try {
        resultElement.textContent = "Predicting...";
        confidenceElement.textContent = "Please wait...";

        const dataURL = canvas.toDataURL("image/png");

        const response = await fetch("/predict", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                image: dataURL
            })
        });

        const data = await response.json();

        // Display backend error instead of undefined.
        if (!response.ok || data.error) {
            resultElement.textContent =
                "Error: " + (data.error || "Prediction failed");

            confidenceElement.textContent = "";

            console.error("Prediction error:", data);
            return;
        }

        if (data.digit === undefined || data.confidence === undefined) {
            throw new Error(
                "The server response is missing digit or confidence."
            );
        }

        // Show predicted digit.
        resultElement.innerHTML = "";

        const label = document.createElement("span");
        label.textContent = "Predicted Digit: ";

        const digitValue = document.createElement("span");
        digitValue.textContent = data.digit;
        digitValue.style.color = "green";
        digitValue.style.fontSize = "40px";

        resultElement.append(label, digitValue);

        // Show confidence.
        confidenceElement.textContent =
            "Confidence Score: " +
            Number(data.confidence).toFixed(2) + "%";

        // Top 3 predictions.
        displayTop3(data.top3);

        // Voice.
        speakPrediction(data.digit);

        // Store this result in the browser's live history.
        history.push({
            id: data.id,
            digit: data.digit,
            confidence: data.confidence
        });

        // Display live history.
        liveHistoryElement.innerHTML = "";

        history.slice(-20).forEach(function (item) {
            const box = document.createElement("span");

            box.textContent = item.digit;
            box.title = "Confidence: " + item.confidence + "%";

            liveHistoryElement.appendChild(box);
        });

        // Update chart.
        if (chart) {
            chart.data.labels = history.map(
                (_, index) => index + 1
            );

            chart.data.datasets[0].data = history.map(
                item => item.digit
            );

            chart.update();
        }

        // Refresh saved database history.
        await loadHistory();

    } catch (error) {
        console.error("Prediction failed:", error);

        resultElement.textContent =
            "Prediction failed. Check the PyCharm terminal.";

        confidenceElement.textContent = "";
    }
}


// ==========================================
// 8. LOAD DATABASE HISTORY
// ==========================================

async function loadHistory() {
    try {
        const response = await fetch("/history");

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Unable to load history.");
        }

        tableElement.innerHTML = "";

        data.forEach(function (row) {
            const tr = document.createElement("tr");

            [row[0], row[1], Number(row[2]).toFixed(2) + "%", row[3]]
                .forEach(function (value) {
                    const td = document.createElement("td");
                    td.textContent = value;
                    tr.appendChild(td);
                });

            tableElement.appendChild(tr);
        });

    } catch (error) {
        console.error("History error:", error);
    }
}


// ==========================================
// 9. LOAD ANALYTICS
// ==========================================

async function loadAnalytics() {
    try {
        const response = await fetch("/analytics");
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Analytics unavailable.");
        }

        // Optional elements: only update them if present in index.html.
        const totalElement = document.getElementById("totalPredictions");

        if (totalElement) {
            totalElement.textContent = data.total_predictions;
        }

        const averageElement = document.getElementById("averageConfidence");

        if (averageElement) {
            averageElement.textContent =
                data.average_confidence.toFixed(2) + "%";
        }

        const accuracyElement = document.getElementById("feedbackAccuracy");

        if (accuracyElement) {
            accuracyElement.textContent =
                data.feedback_accuracy === null
                    ? "No feedback yet"
                    : data.feedback_accuracy + "%";
        }

    } catch (error) {
        console.error("Analytics error:", error);
    }
}


// ==========================================
// 10. SUBMIT USER FEEDBACK
// ==========================================

async function submitFeedback(predictionId, actualDigit) {
    try {
        const response = await fetch("/feedback", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                prediction_id: predictionId,
                actual_digit: actualDigit
            })
        });

        const data = await response.json();

        if (!response.ok) {
            alert(data.error || "Could not save feedback.");
            return;
        }

        alert(data.message);

        await loadAnalytics();

    } catch (error) {
        console.error("Feedback error:", error);
        alert("Could not submit feedback.");
    }
}


// ==========================================
// 11. INITIALIZE PAGE
// ==========================================

document.addEventListener("DOMContentLoaded", function () {
    clearCanvas();
    loadHistory();
    loadAnalytics();
});