// ======================================================
// Adaptive Optics Dashboard
// ======================================================

async function updateDashboard() {
    try {
        const response = await fetch("/data");
        const data = await response.json();

        //---------------------------------------------
        // Frame Counter
        //---------------------------------------------
        document.getElementById("frameCounter").innerHTML =
            "Frame : " + data.frame;

        //---------------------------------------------
        // Shift Table
        //---------------------------------------------
        const shiftBody = document.querySelector("#shiftTable tbody");
        shiftBody.innerHTML = "";

        data.shift_data.forEach((shift, index) => {
            const row = document.createElement("tr");
            row.innerHTML = `
            <td>${index + 1}</td>
            <td>${shift[0].toFixed(3)}</td>
            <td>${shift[1].toFixed(3)}</td>
            `;
            shiftBody.appendChild(row);
        });

        //---------------------------------------------
        // Zernike Table
        //---------------------------------------------
        const zBody = document.querySelector("#zernikeTable tbody");
        zBody.innerHTML = "";

        for (const key in data.zernike) {
            const row = document.createElement("tr");
            row.innerHTML = `
            <td>${key}</td>
            <td>${Number(data.zernike[key]).toFixed(5)}</td>
            `;
            zBody.appendChild(row);
        }

        //---------------------------------------------
        // Turbulence
        //---------------------------------------------
        if (Object.keys(data.turbulence).length > 0) {
            document.getElementById("r0").innerHTML = data.turbulence.r0 + " cm";
            document.getElementById("tau0").innerHTML = data.turbulence.tau0 + " ms";

            const s = document.getElementById("strength");
            s.innerHTML = data.turbulence.strength;
            s.className = "";

            if (data.turbulence.strength == "Weak") {
                s.classList.add("status-good");
            } else if (data.turbulence.strength == "Moderate") {
                s.classList.add("status-medium");
            } else {
                s.classList.add("status-bad");
            }
        }

        //---------------------------------------------
        // DM Commands
        //---------------------------------------------
        const dmTable = document.getElementById("dmTable");
        dmTable.innerHTML = "";

        const values = Object.values(data.dm_commands);
        let index = 0;

        for (let r = 0; r < 8; r++) {
            const row = document.createElement("tr");
            for (let c = 0; c < 8; c++) {
                const td = document.createElement("td");
                
                if (index < values.length) {
                    const v = values[index];
                    td.innerHTML = v.toFixed(2);

                    //---------------------------------
                    // Heatmap Colour
                    //---------------------------------
                    let intensity = Math.min(255, Math.abs(v) * 500);
                    if (v >= 0) {
                        td.style.backgroundColor = `rgb(0,${255 - intensity},255)`;
                    } else {
                        td.style.backgroundColor = `rgb(255,${255 - intensity},0)`;
                    }
                }
                row.appendChild(td);
                index++;
            }
            dmTable.appendChild(row);
        }

        //---------------------------------------------
        // Wavefront Heatmap
        //---------------------------------------------
        drawWavefront(data.wavefront);

        //---------------------------------------------
        // Predicted Next Zernike Coefficients
        //---------------------------------------------
        const predictionBox = document.getElementById('prediction');
        
        if (data.prediction) {
            if (data.prediction.status) {
                // Catches the "Training model..." string from main.py
                predictionBox.innerText = data.prediction.status;
            } else if (data.prediction.error) {
                // Handles other potential prediction errors gracefully
                predictionBox.innerText = "Error: " + data.prediction.error;
            } else {
                // Formats the dictionary nicely
                predictionBox.innerText = JSON.stringify(data.prediction, null, 4);
            }
        } else {
            predictionBox.innerText = "No prediction data available yet.";
        }

    } catch (error) {
        console.error("Error fetching data from API:", error);
    }
}

//==========================================================
// Draw Wavefront
//==========================================================

function drawWavefront(wavefront) {
    const canvas = document.getElementById("wavefrontCanvas");
    const ctx = canvas.getContext("2d");

    const rows = wavefront.length;
    const cols = wavefront[0].length;
    const cellW = canvas.width / cols;
    const cellH = canvas.height / rows;

    let min = 999;
    let max = -999;

    wavefront.forEach(r => {
        r.forEach(v => {
            if (v < min) min = v;
            if (v > max) max = v;
        });
    });

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
            const value = wavefront[y][x];
            const norm = (value - min) / (max - min + 0.00001);

            const red = Math.floor(255 * norm);
            const blue = Math.floor(255 * (1 - norm));

            ctx.fillStyle = `rgb(${red},0,${blue})`;
            ctx.fillRect(x * cellW, y * cellH, cellW, cellH);

            ctx.strokeStyle = "#222";
            ctx.strokeRect(x * cellW, y * cellH, cellW, cellH);
        }
    }
}

//==========================================================
// Initialization
//==========================================================

// Initial call
updateDashboard();

// Update every second
setInterval(updateDashboard, 1000);