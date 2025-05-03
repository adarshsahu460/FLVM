const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const sudo = require('sudo-prompt');
const path = require('path');
const { spawn } = require('child_process');

function createWindow() {
  const win = new BrowserWindow({
    width: 500,
    height: 420,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true
    }
  });
  win.loadFile('./renderer/index.html');
}

app.whenReady().then(createWindow);

ipcMain.handle('select-folder', async () => {
  const result = await dialog.showOpenDialog({ properties: ['openDirectory'] });
  return result.filePaths[0];
});

function sendLog(win, msg) {
  win.webContents.send('log-update', msg);
}

ipcMain.handle('run-client', async (event, { hospital, folder }) => {
  const win = BrowserWindow.getFocusedWindow();
  return new Promise((resolve) => {
    // 1. Create venv if not exists
    sendLog(win, 'Checking/creating virtual environment...');
    sudo.exec('python -m venv venv', { name: 'Federated Client', shell: true }, (err) => {
      if (err) {
        sendLog(win, 'Error creating venv: ' + err.message);
        resolve({ error: err.message });
        return;
      }
      sendLog(win, 'Installing requirements...');
      // 2. Install requirements
      sudo.exec('.\\venv\\Scripts\\activate && pip install -r ..\\backend\\requirements.txt', { name: 'Federated Client', shell: true }, (err2) => {
        if (err2) {
          sendLog(win, 'Error installing requirements: ' + err2.message);
          resolve({ error: err2.message });
          return;
        }
        sendLog(win, 'Starting federated client process...');
        // 3. Run client.py and stream logs
        const clientProc = spawn('.\\venv\\Scripts\\python.exe', ['client.py', '--cid', hospital, '--data-dir', folder], { shell: true });
        clientProc.stdout.on('data', (data) => sendLog(win, data.toString()));
        clientProc.stderr.on('data', (data) => sendLog(win, data.toString()));
        clientProc.on('close', (code) => {
          sendLog(win, `Training finished with exit code ${code}`);
          resolve({ stdout: 'Training finished', stderr: '' });
        });
      });
    });
  });
});