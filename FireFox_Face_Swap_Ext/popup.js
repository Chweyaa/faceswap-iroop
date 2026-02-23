document.getElementById("swapBtn").addEventListener("click", async () => {
    const status = document.getElementById("status");
    const faceIndex = document.getElementById("faceIndex").value;
    
    // 1. Check server status
    try {
        const response = await fetch("http://127.0.0.1:5000/status");
        const data = await response.json();
        
        if (!data.source_selected) {
            status.innerText = "Error: Select Source Face in App!";
            return;
        }
    } catch (e) {
        status.innerText = "Error: App not running!";
        return;
    }

    status.innerText = "Scanning...";

    // 2. Scan page for images
    const foundImages = await browser.tabs.executeScript({
        code: `
            (function() {
                const images = document.querySelectorAll('img');
                let dataList = [];
                let counter = 0;
                
                for (let img of images) {
                    // Skip tiny icons
                    if (img.width < 50 || img.height < 50) continue;
                    
                    // --- KEY FIX FOR RE-SWAPPING ---
                    // If we haven't saved the original URL yet, save it now.
                    // This ensures we always swap from the clean, original source,
                    // not the already-swapped version.
                    if (!img.hasAttribute('data-iroop-original-src')) {
                        img.setAttribute('data-iroop-original-src', img.src);
                    }
                    
                    // We grab the ORIGINAL url, not current src
                    let urlToSend = img.getAttribute('data-iroop-original-src');

                    // Assign ID if missing
                    let uniqueId = img.getAttribute('data-iroop-id');
                    if (!uniqueId) {
                        uniqueId = "iroop_img_" + counter++;
                        img.setAttribute('data-iroop-id', uniqueId);
                    }
                    
                    dataList.push({ id: uniqueId, url: urlToSend });
                }
                return dataList;
            })();
        `
    });

    const imageList = foundImages[0];
    
    if (!imageList || imageList.length === 0) {
        status.innerText = "No images found.";
        return;
    }

    status.innerText = `Processing ${imageList.length} images...`;

    // 3. Process Images
    let successCount = 0;

    for (let item of imageList) {
        try {
            const response = await fetch("http://127.0.0.1:5000/swap", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    url: item.url,
                    face_index: parseInt(faceIndex)
                })
            });

            const data = await response.json();

            if (data.image) {
                const updateCode = `
                    (function() {
                        const img = document.querySelector('img[data-iroop-id="${item.id}"]');
                        if (img) {
                            img.src = "${data.image}";
                            img.removeAttribute('srcset'); // Kill responsive images
                            
                            // Change border color based on Face Index to show it changed
                            const colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF"];
                            const colorIndex = ${faceIndex} + 1; 
                            const color = colors[colorIndex % colors.length] || "#2aa666";
                            
                            img.style.border = "3px solid " + color;
                        }
                    })();
                `;
                await browser.tabs.executeScript({ code: updateCode });
                successCount++;
                status.innerText = `Swapped ${successCount}/${imageList.length}...`;
            }
        } catch (err) {
            console.log("Skipped:", item.url);
        }
    }

    status.innerText = `Done! Swapped ${successCount} images.`;
});