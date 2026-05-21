document.getElementById("scanBtn").addEventListener("click", async () => {

    const [tab] = await chrome.tabs.query({
        active: true,
        currentWindow: true
    });

    chrome.scripting.executeScript({

        target: {
            tabId: tab.id
        },

        func: () => document.body.innerText

    }, async (results) => {

        const text = results[0].result;

        const response = await fetch(
            "http://127.0.0.1:5000/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    text: text
                })
            }
        );

        const data = await response.json();

        document.getElementById("result").innerHTML =
            `
            Prediction: ${data.prediction}<br>
            Confidence: ${data.confidence.toFixed(2)}%
            `;
    });
});