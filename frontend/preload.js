const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  selectFolder: () => ipcRenderer.invoke('select-folder'),
  runClient: (hospital, folder) => ipcRenderer.invoke('run-client', { hospital, folder }),
  receiveLogUpdate: (callback) => ipcRenderer.on('log-update', callback),
});