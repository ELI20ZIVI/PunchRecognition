const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Configurazione multer per upload file
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const uploadDir = path.join(__dirname, 'uploads');
        if (!fs.existsSync(uploadDir)) {
            fs.mkdirSync(uploadDir, { recursive: true });
        }
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        cb(null, `session_${Date.now()}.csv`);
    }
});

const upload = multer({ 
    storage: storage,
    fileFilter: (req, file, cb) => {
        if (file.mimetype === 'text/csv' || file.originalname.endsWith('.csv')) {
            cb(null, true);
        } else {
            cb(new Error('Only CSV files are allowed'), false);
        }
    }
});

// Stato dell'analisi
let analysisState = {
    status: 'idle',
    progress: 0,
    results: null,
    error: null
};

// Route principale
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Upload e analisi del file CSV
app.post('/api/upload', upload.single('csvFile'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
    }

    analysisState = {
        status: 'processing',
        progress: 0,
        results: null,
        error: null,
        filename: req.file.filename
    };

    const csvPath = req.file.path;
    const pythonScript = path.join(__dirname, 'python', 'analizer.py');

    // Avvia lo script Python per l'analisi
    // Usa il Python del venv in dashboard/Code/web_app/python/sportTech
    const venvPython = path.join(__dirname, 'python', 'sportTech', 'bin', 'python');
    const pythonCmd = fs.existsSync(venvPython) ? venvPython : 'python3';

    console.log(`Using Python interpreter: ${pythonCmd}`);

    const pythonProcess = spawn(pythonCmd, [
        pythonScript,
        csvPath
    ]);

    let outputData = '';
    let errorData = '';

    pythonProcess.stdout.on('data', (data) => {
        const output = data.toString();
        outputData += output;
        
        // Parsing del progresso
        const progressMatch = output.match(/PROGRESS:(\d+)/);
        if (progressMatch) {
            analysisState.progress = parseInt(progressMatch[1]);
        }
    });

    pythonProcess.stderr.on('data', (data) => {
        errorData += data.toString();
        console.error('Python stderr:', data.toString());
    });

    pythonProcess.on('close', (code) => {
        if (code === 0) {
            try {
                // Cerca il JSON nei risultati
                const jsonMatch = outputData.match(/RESULTS_JSON:(.*?)END_RESULTS_JSON/s);
                if (jsonMatch) {
                    analysisState.results = JSON.parse(jsonMatch[1]);
                    analysisState.status = 'completed';
                } else {
                    throw new Error('Could not parse results');
                }
            } catch (e) {
                analysisState.status = 'error';
                analysisState.error = 'Failed to parse analysis results';
            }
        } else {
            analysisState.status = 'error';
            analysisState.error = errorData || 'Analysis failed';
        }
    });

    res.json({ 
        message: 'Analysis started',
        filename: req.file.filename 
    });
});

// Stato dell'analisi
app.get('/api/status', (req, res) => {
    res.json(analysisState);
});

// Risultati dell'analisi
app.get('/api/results', (req, res) => {
    if (analysisState.status === 'completed' && analysisState.results) {
        res.json(analysisState.results);
    } else if (analysisState.status === 'error') {
        res.status(500).json({ error: analysisState.error });
    } else {
        res.status(202).json({ message: 'Analysis in progress', progress: analysisState.progress });
    }
});

// Reset dello stato
app.post('/api/reset', (req, res) => {
    analysisState = {
        status: 'idle',
        progress: 0,
        results: null,
        error: null
    };
    res.json({ message: 'State reset' });
});

// Avvio del server
app.listen(PORT, () => {
    console.log(`
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║   🥊 Boxing Punch Analyzer                                 ║
    ║                                                            ║
    ║   Server running at: http://localhost:${PORT}                 ║
    ║                                                            ║
    ║   Ready to analyze your training sessions!                 ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    `);
});