const hospitalInput = document.getElementById('hospital');
const selectFolderBtn = document.getElementById('select-folder');
const folderPathSpan = document.getElementById('folder-path');
const startBtn = document.getElementById('start');
const outputPre = document.getElementById('output');

let selectedFolder = '';

selectFolderBtn.onclick = async () => {
  const folder = await window.electronAPI.selectFolder();
  if (folder) {
    selectedFolder = folder;
    folderPathSpan.textContent = folder;
    folderPathSpan.classList.add('fade-in');
  }
};

startBtn.onclick = async () => {
  const hospital = hospitalInput.value.trim();
  if (!hospital) {
    outputPre.textContent += "Please enter the hospital name.\n";
    return;
  }
  if (!selectedFolder) {
    outputPre.textContent += "Please select a folder with images.\n";
    return;
  }
  outputPre.textContent += "Running training... (You may see an admin prompt)\n";
  startBtn.disabled = true;
  const result = await window.electronAPI.runClient(hospital, selectedFolder);
  if (result.error) {
    outputPre.textContent += "Error: " + result.error + "\n" + (result.stderr || "") + "\n";
  } else {
    outputPre.textContent += (result.stdout || "") + (result.stderr || "") + "\n";
  }
  startBtn.disabled = false;
};

window.electronAPI.receiveLogUpdate((event, msg) => {
  outputPre.textContent += msg + '\n';
});