let podcastUrl = null;
let isPlaying = false;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let pauseTime = 0;

function handleFileUpload(event) {
    const file = event.target.files[0] || event.dataTransfer.files[0];
    if (file && file.type === "application/pdf") {
        document.getElementById("file-name").textContent = file.name;
        uploadPdf(file);
    } else {
        alert("Please upload a valid PDF file.");
    }
}

function handleDrop(event) {
    event.preventDefault();
    handleFileUpload(event);
}

async function uploadPdf(file) {
    const formData = new FormData();
    formData.append("pdf", file);

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData,
        });
        if (!response.ok) throw new Error(`Upload failed: ${response.status}`);
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        podcastUrl = data.podcastUrl;
        setupPodcastPlayer();
    } catch (error) {
        console.error("Upload failed:", error);
        alert("Failed to generate podcast. Try again!");
    }
}

function setupPodcastPlayer() {
    const playbackSection = document.getElementById("playback-section");
    const uploadPrompt = document.getElementById("upload-prompt");
    const audio = document.getElementById("podcast-audio");

    audio.src = podcastUrl;
    playbackSection.classList.remove("hidden");
    uploadPrompt.classList.add("hidden");

    audio.addEventListener("error", (e) => {
        console.error("Audio loading error:", e);
        alert("Error loading podcast audio.");
    });

    audio.addEventListener("timeupdate", () => {
        const progress = document.querySelector("progress");
        const percent = (audio.currentTime / audio.duration) * 100;
        progress.value = percent || 0;
    });

    audio.addEventListener("canplay", () => {
        document.getElementById("play-btn").disabled = false;
    });
}

function togglePlay() {
    const audio = document.getElementById("podcast-audio");
    const playBtn = document.getElementById("play-btn");

    if (!audio) return console.error("Audio element not found");

    if (isPlaying && !isRecording) {
        audio.pause();
    } else {
        audio.play().catch((error) => {
            console.error("Playback failed:", error);
            alert("Failed to play podcast.");
        });
    }
    isPlaying = !isPlaying;
    playBtn.textContent = isPlaying ? "Pause" : "Play";
}

function downloadPodcast() {
    if (podcastUrl) {
        const link = document.createElement("a");
        link.href = podcastUrl;
        link.download = "podcast.mp3";
        link.click();
    } else {
        alert("No podcast available to download.");
    }
}

async function toggleRecording() {
    const micBtn = document.getElementById("mic-btn");
    const micStatus = document.getElementById("mic-status");
    const recordingStatus = document.getElementById("recording-status");
    const podcastAudio = document.getElementById("podcast-audio");

    if (!isRecording) {
        // Start recording
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = (event) => audioChunks.push(event.data);
            mediaRecorder.onstop = sendAudioToBackend;

            mediaRecorder.start();
            isRecording = true;

            // Pause podcast and store pause time
            if (isPlaying) {
                pauseTime = podcastAudio.currentTime;
                podcastAudio.pause();
                isPlaying = false;
                document.getElementById("play-btn").textContent = "Play";
            }

            micStatus.textContent = "Mic On";
            micBtn.classList.add("recording");
            recordingStatus.classList.remove("hidden");
        } catch (error) {
            console.error("Microphone access failed:", error);
            alert("Failed to access microphone. Please allow permission.");
        }
    } else {
        // Stop recording
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        isRecording = false;

        micStatus.textContent = "Ask a Question (Mic Off)";
        micBtn.classList.remove("recording");
        recordingStatus.classList.add("hidden");
    }
}

async function sendAudioToBackend() {
    const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
    const formData = new FormData();
    formData.append("audio", audioBlob, "question.webm");

    try {
        const response = await fetch("/ask-question", {
            method: "POST",
            body: formData,
        });
        if (!response.ok) throw new Error("Failed to process question");
        const data = await response.json();

        // Play the response audio
        const responseAudio = document.getElementById("response-audio");
        responseAudio.src = data.responseUrl;
        responseAudio.classList.remove("hidden");
        responseAudio.play().catch((error) => {
            console.error("Response playback failed:", error);
            alert("Failed to play response.");
        });

        // Resume podcast 3 seconds after response ends
        responseAudio.onended = () => {
            responseAudio.classList.add("hidden");
            const podcastAudio = document.getElementById("podcast-audio");
            if (podcastUrl) {
                podcastAudio.currentTime = Math.max(0, pauseTime - 3); // Rewind 3 seconds
                setTimeout(() => {
                    podcastAudio.play();
                    isPlaying = true;
                    document.getElementById("play-btn").textContent = "Pause";
                }, 3000); // Wait 3 seconds before resuming
            }
        };
    } catch (error) {
        console.error("Error sending audio to backend:", error);
        alert("Failed to process your question.");
        resumePodcast(); // Resume anyway if error occurs
    }
}

function resumePodcast() {
    const podcastAudio = document.getElementById("podcast-audio");
    if (podcastUrl) {
        podcastAudio.currentTime = Math.max(0, pauseTime - 3);
        setTimeout(() => {
            podcastAudio.play();
            isPlaying = true;
            document.getElementById("play-btn").textContent = "Pause";
        }, 3000);
    }
}