chrome.runtime.onInstalled.addListener(() => {

    chrome.contextMenus.create({

        id: "scan-ai",

        title: "Detect AI Content",

        contexts: ["selection"]
    });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {

    if (info.menuItemId === "scan-ai") {

        try {

            const selectedText = info.selectionText;

            const response = await fetch(
                "http://127.0.0.1:5000/predict",
                {
                    method: "POST",

                    headers: {
                        "Content-Type": "application/json"
                    },

                    body: JSON.stringify({
                        text: selectedText
                    })
                }
            );

            const data = await response.json();

            chrome.scripting.executeScript({

                target: {
                    tabId: tab.id
                },

                func: (result) => {

                    alert(
                        `Prediction: ${result.prediction}\n` +
                        `Confidence: ${result.confidence.toFixed(2)}%`
                    );
                },

                args: [data]
            });

        } catch (error) {

            console.error("Fetch Error:", error);
        }
    }
});