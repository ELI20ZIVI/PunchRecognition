/**
 * Boxing Punch Analyzer - Frontend Application
 */

class BoxingAnalyzer {
    constructor() {
        this.analysisResults = null;
        this.gaugeCharts = {};
        this.timelineChart = null;
        this.punchDetailChart = null;
        
        this.init();
    }
    
    init() {
        this.bindElements();
        this.bindEvents();
    }
    
    bindElements() {
        // Sections
        this.uploadSection = document.getElementById('uploadSection');
        this.processingSection = document.getElementById('processingSection');
        this.resultsSection = document.getElementById('resultsSection');
        
        // Upload elements
        this.csvInput = document.getElementById('csvInput');
        this.uploadArea = document.getElementById('uploadArea');
        this.selectFileBtn = document.getElementById('selectFileBtn');
        this.fileInfo = document.getElementById('fileInfo');
        this.fileName = document.getElementById('fileName');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        
        // Processing elements
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');
        this.processingInfo = document.getElementById('processingInfo');
        
        // Results elements
        this.totalPunches = document.getElementById('totalPunches');
        this.sessionDuration = document.getElementById('sessionDuration');
        this.punchesPerMin = document.getElementById('punchesPerMin');
        this.newSessionBtn = document.getElementById('newSessionBtn');
        
        // Modals
        this.punchModal = document.getElementById('punchModal');
        this.modalOverlay = document.getElementById('modalOverlay');
        this.modalClose = document.getElementById('modalClose');
        this.modalTitle = document.getElementById('modalTitle');
        this.punchTableBody = document.getElementById('punchTableBody');
        
        this.punchDetailModal = document.getElementById('punchDetailModal');
        this.detailModalOverlay = document.getElementById('detailModalOverlay');
        this.detailModalClose = document.getElementById('detailModalClose');
    }
    
    bindEvents() {
        // File selection
        this.selectFileBtn.addEventListener('click', () => this.csvInput.click());
        this.csvInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Drag and drop
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        
        // Analyze button
        this.analyzeBtn.addEventListener('click', () => this.startAnalysis());
        
        // New session
        this.newSessionBtn.addEventListener('click', () => this.resetSession());
        
        // Modal events
        this.modalOverlay.addEventListener('click', () => this.closePunchModal());
        this.modalClose.addEventListener('click', () => this.closePunchModal());
        this.detailModalOverlay.addEventListener('click', () => this.closePunchDetailModal());
        this.detailModalClose.addEventListener('click', () => this.closePunchDetailModal());
        
        // Gauge card clicks
        document.querySelectorAll('.gauge-card').forEach(card => {
            card.addEventListener('click', () => {
                const punchType = card.dataset.punchType;
                this.openPunchModal(punchType);
            });
        });
    }
    
    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.selectedFile = file;
            this.showFileInfo(file.name);
        }
    }
    
    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('drag-over');
    }
    
    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('drag-over');
    }
    
    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('drag-over');
        
        const file = e.dataTransfer.files[0];
        if (file && file.name.endsWith('.csv')) {
            this.selectedFile = file;
            this.showFileInfo(file.name);
        }
    }
    
    showFileInfo(name) {
        this.fileName.textContent = name;
        this.fileInfo.style.display = 'flex';
    }
    
    async startAnalysis() {
        if (!this.selectedFile) return;
        
        // Show processing section
        this.uploadSection.style.display = 'none';
        this.processingSection.style.display = 'flex';
        
        // Upload file
        const formData = new FormData();
        formData.append('csvFile', this.selectedFile);
        
        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) throw new Error('Upload failed');
            
            // Poll for status
            this.pollStatus();
            
        } catch (error) {
            console.error('Error:', error);
            this.showError('Upload fallito. Riprova.');
        }
    }
    
    async pollStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if (data.status === 'processing') {
                this.updateProgress(data.progress);
                setTimeout(() => this.pollStatus(), 500);
            } else if (data.status === 'completed') {
                this.updateProgress(100);
                
                console.log('Risultati ricevuti:', data.results);
                console.log('Punches prima normalizzazione:', data.results.punches?.map(p => p.type));
                
                this.analysisResults = data.results;
                
                // Normalizza i tipi di punch ricevuti dal backend
                if (this.analysisResults.punches) {
                    this.analysisResults.punches = this.analysisResults.punches.map(punch => ({
                        ...punch,
                        type: this.normalizePunchType(punch.type)
                    }));
                }
                
                // Normalizza anche punchCounts
                if (this.analysisResults.punchCounts) {
                    const normalizedCounts = {};
                    Object.keys(this.analysisResults.punchCounts).forEach(key => {
                        normalizedCounts[this.normalizePunchType(key)] = this.analysisResults.punchCounts[key];
                    });
                    this.analysisResults.punchCounts = normalizedCounts;
                }
                
                console.log('Punches dopo normalizzazione:', this.analysisResults.punches.map(p => p.type));
                console.log('PunchCounts normalizzati:', this.analysisResults.punchCounts);
                
                setTimeout(() => this.showResults(), 500);
            } else if (data.status === 'error') {
                this.showError(data.error);
            }
        } catch (error) {
            console.error('Errore polling:', error);
            this.showError('Errore durante l\'analisi');
        }
    }
    
    updateProgress(progress) {
        this.progressFill.style.width = `${progress}%`;
        this.progressText.textContent = `${progress}%`;
        
        if (progress < 30) {
            this.processingInfo.textContent = 'Caricamento dati sensori...';
        } else if (progress < 60) {
            this.processingInfo.textContent = 'Rilevamento colpi...';
        } else if (progress < 90) {
            this.processingInfo.textContent = 'Analisi metriche...';
        } else {
            this.processingInfo.textContent = 'Finalizzazione risultati...';
        }
    }
    
    showResults() {
        this.processingSection.style.display = 'none';
        this.resultsSection.style.display = 'block';
        
        if (!this.analysisResults) return;
        
        // Update summary stats
        this.totalPunches.textContent = this.analysisResults.totalPunches;
        this.sessionDuration.textContent = this.formatDuration(this.analysisResults.duration);
        this.punchesPerMin.textContent = this.analysisResults.punchesPerMinute.toFixed(1);
        
        // Create gauge charts
        this.createGaugeCharts();
        
        // Create timeline
        this.createTimeline();
    }
    
    normalizePunchType(punchType) {
        // Mapping solo dei tipi effettivi usati dal modello ML
        const mappings = {
            // Formato Lead/Rear dal modello (Lead = mano avanti, Rear = mano dietro)
            'Lead_hook': 'hook_left',
            'Rear_hook': 'hook_right',
            'Lead_jab': 'jab_left',
            'Rear_jab': 'jab_right',
            'Lead_uppercut': 'uppercut_left',
            'Rear_uppercut': 'uppercut_right',
            // Guard
            'guard_noPunches': 'guard',
        };
        
        // Cerca nel mapping esplicito
        if (mappings[punchType]) {
            return mappings[punchType];
        }
        
        // Fallback: converti a lowercase, normalizza spazi/underscore
        const normalized = punchType
            .toLowerCase()
            .replace(/\s+/g, '_')
            .replace(/([a-z])([A-Z])/g, '$1_$2')
            .toLowerCase();
        
        return normalized;
    }
    
    createGaugeCharts() {
        const punchTypes = [
            { id: 'jab_left', canvasId: 'gaugeJabLeft', countId: 'countJabLeft', color: '#4cc9f0' },
            { id: 'jab_right', canvasId: 'gaugeJabRight', countId: 'countJabRight', color: '#4cc9f0' },
            { id: 'hook_left', canvasId: 'gaugeHookLeft', countId: 'countHookLeft', color: '#f72585' },
            { id: 'hook_right', canvasId: 'gaugeHookRight', countId: 'countHookRight', color: '#f72585' },
            { id: 'uppercut_left', canvasId: 'gaugeUppercutLeft', countId: 'countUppercutLeft', color: '#7209b7' },
            { id: 'uppercut_right', canvasId: 'gaugeUppercutRight', countId: 'countUppercutRight', color: '#7209b7' }
        ];
        
        punchTypes.forEach(punch => {
            const count = this.analysisResults.punchCounts[punch.id] || 0;
            const percentage = this.analysisResults.totalPunches > 0 
                ? (count / this.analysisResults.totalPunches) * 100 
                : 0;
            
            // Update count display
            document.getElementById(punch.countId).textContent = count;
            
            // Create gauge chart
            const canvas = document.getElementById(punch.canvasId);
            const ctx = canvas.getContext('2d');
            
            // Destroy existing chart
            if (this.gaugeCharts[punch.id]) {
                this.gaugeCharts[punch.id].destroy();
            }
            
            this.gaugeCharts[punch.id] = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    datasets: [{
                        data: [percentage, 100 - percentage],
                        backgroundColor: [punch.color, '#1a1a2e'],
                        borderWidth: 0
                    }]
                },
                options: {
                    cutout: '75%',
                    rotation: -90,
                    circumference: 180,
                    plugins: {
                        legend: { display: false },
                        tooltip: { enabled: false }
                    },
                    animation: {
                        animateRotate: true,
                        duration: 1000
                    }
                },
                plugins: [{
                    id: 'centerText',
                    afterDraw: (chart) => {
                        const ctx = chart.ctx;
                        const centerX = chart.width / 2;
                        const centerY = chart.height - 20;
                        
                        ctx.save();
                        ctx.font = 'bold 20px Oswald';
                        ctx.fillStyle = punch.color;
                        ctx.textAlign = 'center';
                        ctx.fillText(`${percentage.toFixed(0)}%`, centerX, centerY);
                        ctx.restore();
                    }
                }]
            });
        });
    }
    
    createTimeline() {
        const container = document.getElementById('timelineChart');
        container.innerHTML = '';
        
        if (!this.analysisResults.punches || this.analysisResults.punches.length === 0) {
            container.innerHTML = '<p style="text-align: center; padding: 2rem; color: #666;">Nessun colpo rilevato</p>';
            return;
        }
        
        const canvas = document.createElement('canvas');
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        container.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        
        // Prepare timeline data
        const punches = this.analysisResults.punches;
        const duration = this.analysisResults.duration;
        
        console.log('Timeline - Tutti i punch:', punches.map(p => ({ type: p.type, timestamp: p.timestamp })));
        
        // Group punches by type
        const punchColors = {
            'jab_left': '#4361ee',
            'jab_right': '#e63946',
            'hook_left': '#f72585',
            'hook_right': '#ff6b6b',
            'uppercut_left': '#7209b7',
            'uppercut_right': '#9d4edd'
        };
        
        const datasets = Object.keys(punchColors).map(type => {
            const matchingPunches = punches.filter(p => p.type === type);
            console.log(`Timeline - Tipo "${type}": ${matchingPunches.length} colpi trovati`);
            return {
                label: type.replace('_', ' ').toUpperCase(),
                data: matchingPunches.map(p => ({ x: p.timestamp, y: 1 })),
                backgroundColor: punchColors[type],
                pointRadius: 8,
                pointHoverRadius: 12
            };
        });
        
        if (this.timelineChart) {
            this.timelineChart.destroy();
        }
        
        this.timelineChart = new Chart(ctx, {
            type: 'scatter',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        title: {
                            display: true,
                            text: 'Tempo (secondi)',
                            color: '#a0a0a0'
                        },
                        min: 0,
                        max: duration,
                        grid: { color: '#333355' },
                        ticks: { color: '#a0a0a0' }
                    },
                    y: {
                        display: false,
                        min: 0,
                        max: 2
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#ffffff' }
                    }
                }
            }
        });
    }
    
    openPunchModal(punchType) {
        if (!this.analysisResults) return;
        
        const typeMap = {
            'jab_left': 'Diretto (Sinistro)',
            'jab_right': 'Diretto (Destro)',
            'hook_left': 'Gancio (Sinistro)',
            'hook_right': 'Gancio (Destro)',
            'uppercut_left': 'Montante (Sinistro)',
            'uppercut_right': 'Montante (Destro)'
        };
        
        this.modalTitle.textContent = typeMap[punchType] || punchType;
        
        // Filter punches by type
        const punches = this.analysisResults.punches.filter(p => p.type === punchType);
        
        // Populate table
        this.punchTableBody.innerHTML = '';
        punches.forEach((punch, index) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${this.formatTime(punch.timestamp)}</td>
                <td>${(punch.confidence * 100).toFixed(1)}%</td>
                <td>${punch.peakVelocity.toFixed(2)} m/s</td>
                <td>${punch.duration.toFixed(0)} ms</td>
            `;
            row.addEventListener('click', () => this.openPunchDetail(punch, index + 1));
            this.punchTableBody.appendChild(row);
        });
        
        this.punchModal.classList.add('active');
    }
    
    closePunchModal() {
        this.punchModal.classList.remove('active');
    }
    
    openPunchDetail(punch, index) {
        document.getElementById('detailModalTitle').textContent = `Colpo #${index} - ${punch.type}`;
        document.getElementById('detailTimestamp').textContent = this.formatTime(punch.timestamp);
        document.getElementById('detailConfidence').textContent = `${(punch.confidence * 100).toFixed(1)}%`;
        document.getElementById('detailVelocity').textContent = `${punch.peakVelocity.toFixed(2)} m/s`;
        document.getElementById('detailDuration').textContent = `${punch.duration.toFixed(0)} ms`;
        document.getElementById('detailAcceleration').textContent = `${punch.peakAcceleration.toFixed(2)} g`;
        document.getElementById('detailRotation').textContent = `${punch.peakRotation.toFixed(1)} °/s`;
        
        // Create punch data chart
        this.createPunchDataChart(punch);
        
        this.punchDetailModal.classList.add('active');
    }
    
    closePunchDetailModal() {
        this.punchDetailModal.classList.remove('active');
    }
    
    createPunchDataChart(punch) {
        const ctx = document.getElementById('punchDataChart').getContext('2d');
        
        if (this.punchDetailChart) {
            this.punchDetailChart.destroy();
        }
        
        const sensorData = punch.sensorData || [];
        const labels = sensorData.map((_, i) => i);
        
        this.punchDetailChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    {
                        label: 'Acc X',
                        data: sensorData.map(d => d[0] || 0),
                        borderColor: '#e63946',
                        tension: 0.4,
                        pointRadius: 0
                    },
                    {
                        label: 'Acc Y',
                        data: sensorData.map(d => d[1] || 0),
                        borderColor: '#4cc9f0',
                        tension: 0.4,
                        pointRadius: 0
                    },
                    {
                        label: 'Acc Z',
                        data: sensorData.map(d => d[2] || 0),
                        borderColor: '#f4a261',
                        tension: 0.4,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: { display: true, text: 'Frame', color: '#a0a0a0' },
                        grid: { color: '#333355' },
                        ticks: { color: '#a0a0a0' }
                    },
                    y: {
                        title: { display: true, text: 'Accelerazione (g)', color: '#a0a0a0' },
                        grid: { color: '#333355' },
                        ticks: { color: '#a0a0a0' }
                    }
                },
                plugins: {
                    legend: { labels: { color: '#ffffff' } }
                }
            }
        });
    }
    
    formatDuration(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
    
    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = (seconds % 60).toFixed(2);
        return `${mins}:${parseFloat(secs).toFixed(2).padStart(5, '0')}`;
    }
    
    async resetSession() {
        await fetch('/api/reset', { method: 'POST' });
        
        this.analysisResults = null;
        this.selectedFile = null;
        
        // Reset UI
        this.resultsSection.style.display = 'none';
        this.processingSection.style.display = 'none';
        this.uploadSection.style.display = 'block';
        this.fileInfo.style.display = 'none';
        this.csvInput.value = '';
        
        // Destroy charts
        Object.values(this.gaugeCharts).forEach(chart => chart?.destroy());
        this.gaugeCharts = {};
        this.timelineChart?.destroy();
        this.punchDetailChart?.destroy();
    }
    
    showError(message) {
        alert('Errore: ' + message);
        this.resetSession();
    }
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    window.boxingAnalyzer = new BoxingAnalyzer();
});