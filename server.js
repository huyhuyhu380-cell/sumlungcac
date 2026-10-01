import fastify from "fastify";
import cors from "@fastify/cors";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import fetch from "node-fetch";

// --- CẤU HÌNH ---
const PORT = 3000;
const API_URL = "https://wtxmd52.tele68.com/v1/txmd5/sessions";

// --- GLOBAL STATE ---
let txHistory = [];
let currentSessionId = null;
let fetchInterval = null;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// =====================================================================
// UTILITIES
// =====================================================================
function parseLines(data) {
    if (!data || !Array.isArray(data.list)) return [];
    const sortedList = data.list.sort((a, b) => b.id - a.id);
    const arr = sortedList.map(item => ({
        session: item.id,
        dice: item.dices,
        total: item.point,
        result: item.resultTruyenThong,
        tx: item.point >= 11 ? 'T' : 'X'
    }));
    return arr.sort((a, b) => a.session - b.session);
}

function lastN(arr, n) { return arr.slice(Math.max(0, arr.length - n)); }
function sum(nums) { return nums.reduce((a, b) => a + b, 0); }
function avg(nums) { return nums.length ? sum(nums) / nums.length : 0; }

function majority(obj) {
    let maxK = null, maxV = -Infinity;
    for (const k in obj) if (obj[k] > maxV) { maxV = obj[k]; maxK = k; }
    return { key: maxK, val: maxV };
}

function entropy(arr) {
    if (!arr.length) return 0;
    const freq = {};
    for (const v of arr) freq[v] = (freq[v] || 0) + 1;
    let e = 0, n = arr.length;
    for (const k in freq) { const p = freq[k] / n; e -= p * Math.log2(p); }
    return e;
}

function similarity(a, b) {
    if (a.length !== b.length) return 0;
    let m = 0;
    for (let i = 0; i < a.length; i++) if (a[i] === b[i]) m++;
    return m / a.length;
}
// =====================================================================
// PREDICTION TRACKER (✅/❌) — Không cần load file
// =====================================================================
class PredictionTracker {
    constructor() {
        this.records = new Map();
        this.totalPred = 0;
        this.correctPred = 0;
        this.algoStats = {};
        this.recentResults = [];
    }

    recordPrediction(session, prediction, confidence, algorithmVotes = {}) {
        const existing = this.records.get(session);
        if (existing) {
            existing.predicted = prediction;
            existing.confidence = confidence;
            existing.algorithms = algorithmVotes;
        } else {
            this.records.set(session, {
                session,
                predicted: prediction,
                actual: null,
                correct: null,
                confidence,
                algorithms: algorithmVotes
            });
        }
    }

    recordActual(session, actualTx) {
        const r = this.records.get(session);
        if (!r || !r.predicted) return null;
        r.actual = actualTx;
        r.correct = r.predicted === actualTx;
        this.totalPred++;
        if (r.correct) this.correctPred++;

        const icon = r.correct ? "✅" : "❌";
        this.recentResults.push(icon);
        if (this.recentResults.length > 200) this.recentResults.shift();

        for (const [algoId, pred] of Object.entries(r.algorithms || {})) {
            if (!this.algoStats[algoId]) this.algoStats[algoId] = { total: 0, correct: 0 };
            this.algoStats[algoId].total++;
            if (pred === actualTx) this.algoStats[algoId].correct++;
        }
        return r;
    }

    getRecentResults(n = 20) { return this.recentResults.slice(-n); }

    getOverallAccuracy() {
        if (this.totalPred === 0) return 0;
        return this.correctPred / this.totalPred;
    }

    getAlgoStats() {
        const out = {};
        for (const [id, s] of Object.entries(this.algoStats)) {
            out[id] = {
                total: s.total,
                correct: s.correct,
                accuracy: s.total > 0 ? (s.correct / s.total * 100).toFixed(1) + "%" : "0%"
            };
        }
        return out;
    }
}
// =====================================================================
// FEATURE EXTRACTION
// =====================================================================
function extractFeatures(history) {
    const tx = history.map(h => h.tx);
    const totals = history.map(h => h.total);

    const freq = {};
    for (const v of tx) freq[v] = (freq[v] || 0) + 1;

    let runs = [], cur = tx[0], len = 1;
    for (let i = 1; i < tx.length; i++) {
        if (tx[i] === cur) len++;
        else { runs.push({ val: cur, len }); cur = tx[i]; len = 1; }
    }
    if (tx.length) runs.push({ val: cur, len });

    const meanTotal = avg(totals);
    const variance = avg(totals.map(t => Math.pow(t - meanTotal, 2)));
    const mean5 = avg(totals.slice(-5));
    const mean10 = avg(totals.slice(-10));
    const mean20 = avg(totals.slice(-20));

    let upStreak = 0, downStreak = 0;
    for (let i = totals.length - 1; i > 0; i--) {
        if (totals[i] > totals[i - 1]) upStreak++;
        else break;
    }
    for (let i = totals.length - 1; i > 0; i--) {
        if (totals[i] < totals[i - 1]) downStreak++;
        else break;
    }

    let tStreak = 0, xStreak = 0;
    for (let i = tx.length - 1; i >= 0; i--) {
        if (tx[i] === 'T') tStreak++;
        else break;
    }
    for (let i = tx.length - 1; i >= 0; i--) {
        if (tx[i] === 'X') xStreak++;
        else break;
    }

    let highStreak = 0, lowStreak = 0;
    for (let i = totals.length - 1; i >= 0; i--) {
        if (totals[i] >= 11) highStreak++;
        else break;
    }
    for (let i = totals.length - 1; i >= 0; i--) {
        if (totals[i] <= 10) lowStreak++;
        else break;
    }

    const volatility = Math.sqrt(variance);

    return {
        tx, totals, freq, runs,
        maxRun: runs.reduce((m, r) => Math.max(m, r.len), 0),
        meanTotal, stdTotal: volatility, volatility,
        entropy: entropy(tx),
        last3Pattern: tx.slice(-3).join(''),
        last5Pattern: tx.slice(-5).join(''),
        last8Pattern: tx.slice(-8).join(''),
        last10Pattern: tx.slice(-10).join(''),
        last12Pattern: tx.slice(-12).join(''),
        mean5, mean10, mean20,
        upStreak, downStreak,
        tStreak, xStreak,
        highStreak, lowStreak,
        lastRun: runs[runs.length - 1],
        prevRun: runs[runs.length - 2],
        prev2Run: runs[runs.length - 3],
        prev3Run: runs[runs.length - 4]
    };
}

// =====================================================================
// PATTERN DETECTION (35+ dạng cầu — từ 13 file .txt)
// =====================================================================
function detectPatternType(runs) {
    if (!runs || runs.length < 3) return null;
    const lastRuns = runs.slice(-10);
    const lengths = lastRuns.map(r => r.len);
    const values = lastRuns.map(r => r.val);

    if (lengths.length >= 5 && lengths.slice(-5).every(l => l === 1)) {
        const vals = values.slice(-6);
        if (vals.every((v, i) => i === 0 || v !== vals[i - 1])) return '1_1_pattern';
    }
    if (lengths.length >= 4 && lengths.slice(-4).every(l => l === 2)) {
        const vals = values.slice(-5);
        if (vals.every((v, i) => i === 0 || v !== vals[i - 1])) return '2_2_pattern';
    }
    if (lengths.length >= 3 && lengths.slice(-3).every(l => l === 3)) {
        const vals = values.slice(-4);
        if (vals.every((v, i) => i === 0 || v !== vals[i - 1])) return '3_3_pattern';
    }
    if (lengths.length >= 3 && lengths.slice(-3).every(l => l === 4)) {
        const vals = values.slice(-4);
        if (vals.every((v, i) => i === 0 || v !== vals[i - 1])) return '4_4_pattern';
    }

    if (lengths.length >= 5) {
        const t = lengths.slice(-5);
        if (t[0] === 1 && t[1] === 2 && t[2] === 1 && t[3] === 2 && t[4] === 1) return '1_2_1_pattern';
        if (t[0] === 2 && t[1] === 1 && t[2] === 2 && t[3] === 1 && t[4] === 2) return '2_1_2_pattern';
        if (t[0] === 3 && t[1] === 2 && t[2] === 3 && t[3] === 2 && t[4] === 3) return '3_2_3_pattern';
        if (t[0] === 2 && t[1] === 3 && t[2] === 2 && t[3] === 3 && t[4] === 2) return '2_3_2_pattern';
        if (t[0] === 4 && t[1] === 2 && t[2] === 4 && t[3] === 2 && t[4] === 4) return '4_2_4_pattern';
        if (t[0] === 1 && t[1] === 3 && t[2] === 1 && t[3] === 3 && t[4] === 1) return '1_3_1_pattern';
        if (t[0] === 3 && t[1] === 1 && t[2] === 3 && t[3] === 1 && t[4] === 3) return '3_1_3_pattern';
        if (t[0] === 1 && t[1] === 4 && t[2] === 1 && t[3] === 4 && t[4] === 1) return '1_4_1_pattern';
        if (t[0] === 2 && t[1] === 2 && t[2] === 1 && t[3] === 2 && t[4] === 2) return '2_2_1_pattern';
        if (t[0] === 1 && t[1] === 1 && t[2] === 2 && t[3] === 1 && t[4] === 1) return '1_1_2_pattern';
        if (t[0] === 1 && t[1] === 2 && t[2] === 2 && t[3] === 1 && t[4] === 2) return '1_2_2_pattern';
        if (t[0] === 2 && t[1] === 1 && t[2] === 1 && t[3] === 2 && t[4] === 1) return '2_1_1_pattern';
        if (t[0] === 3 && t[1] === 3 && t[2] === 1 && t[3] === 3 && t[4] === 3) return '3_3_1_pattern';
        if (t[0] === 1 && t[1] === 1 && t[2] === 3 && t[3] === 1 && t[4] === 1) return '1_1_3_pattern';
    }

    if (lengths.length >= 6) {
        const t = lengths.slice(-6);
        if (t[0] === 1 && t[1] === 2 && t[2] === 1 && t[3] === 2 && t[4] === 1 && t[5] === 2) return '1_2_1_2_1_2_pattern';
        if (t[0] === 2 && t[1] === 1 && t[2] === 2 && t[3] === 1 && t[4] === 2 && t[5] === 1) return '2_1_2_1_2_1_pattern';
        if (t[0] === 2 && t[1] === 2 && t[2] === 1 && t[3] === 2 && t[4] === 2 && t[5] === 1) return '2_2_1_loop_pattern';
        if (t[0] === 1 && t[1] === 1 && t[2] === 2 && t[3] === 1 && t[4] === 1 && t[5] === 2) return '1_1_2_loop_pattern';
    }

    if (lengths.length >= 7) {
        const t = lengths.slice(-7);
        if (t[0] === 1 && t[1] === 2 && t[2] === 3 && t[3] === 1 && t[4] === 2 && t[5] === 3 && t[6] === 1) return 'fibonacci_pattern';
        if (t[0] === 3 && t[1] === 2 && t[2] === 1 && t[3] === 3 && t[4] === 2 && t[5] === 1 && t[6] === 3) return 'reverse_fibonacci_pattern';
        if (t[0] === 1 && t[1] === 1 && t[2] === 2 && t[3] === 3 && t[4] === 1 && t[5] === 1 && t[6] === 2) return 'growth_pattern';
    }

    const lastRun = lastRuns[lastRuns.length - 1];
    if (lastRun) {
        if (lastRun.len >= 10) return 'super_long_run_pattern';
        if (lastRun.len >= 7) return 'long_run_pattern';
        if (lastRun.len >= 5) return 'medium_long_run_pattern';
        if (lastRun.len >= 4) return 'medium_run_pattern';
        if (lastRun.len >= 3) return 'short_run_pattern';
    }

    if (lengths.length >= 4) {
        const t = lengths.slice(-4);
        if (t[0] === 2 && t[1] === 1 && t[2] === 3 && t[3] === 2) return 'broken_2_1_3_2_pattern';
        if (t[0] === 3 && t[1] === 1 && t[2] === 2 && t[3] === 3) return 'broken_3_1_2_3_pattern';
        if (t[0] === 2 && t[1] === 3 && t[2] === 1 && t[3] === 2) return 'broken_2_3_1_2_pattern';
        if (t[0] === 1 && t[1] === 3 && t[2] === 2 && t[3] === 1) return 'broken_1_3_2_1_pattern';
    }

    if (lengths.length >= 5) {
        const t = lengths.slice(-5);
        if (t[0] === 1 && t[1] === 2 && t[2] === 3 && t[3] === 2 && t[4] === 1) return 'mirror_pattern_12321';
        if (t[0] === 2 && t[1] === 3 && t[2] === 2 && t[3] === 3 && t[4] === 2) return 'alternating_mirror_pattern';
    }

    if (lengths.length >= 6) {
        const t = lengths.slice(-6);
        if (t[0] === 1 && t[1] === 1 && t[2] === 2 && t[3] === 2 && t[4] === 3 && t[5] === 3) return 'staircase_up_pattern';
        if (t[0] === 3 && t[1] === 3 && t[2] === 2 && t[3] === 2 && t[4] === 1 && t[5] === 1) return 'staircase_down_pattern';
    }

    if (lengths.length >= 4) {
        const t = lengths.slice(-4);
        if (t[0] === 2 && t[1] === 2 && t[2] === 2 && t[3] === 1) return '2_2_2_1_pattern';
        if (t[0] === 2 && t[1] === 2 && t[2] === 1 && t[3] === 2) return '2_2_1_2_pattern';
        if (t[0] === 2 && t[1] === 1 && t[2] === 2 && t[3] === 2) return '2_1_2_2_pattern';
        if (t[0] === 1 && t[1] === 2 && t[2] === 2 && t[3] === 2) return '1_2_2_2_pattern';
    }

    return 'random_pattern';
}

// =====================================================================
// PREDICT FROM PATTERN
// =====================================================================
function predictNextFromPattern(patternType, runs, lastTx) {
    if (!patternType) return null;
    const lastRun = runs[runs.length - 1];
    if (!lastRun) return null;
    const flip = (v) => v === 'T' ? 'X' : 'T';

    switch (patternType) {
        case '1_1_pattern': return flip(lastTx);
        case '2_2_pattern': return lastRun.len === 2 ? flip(lastRun.val) : lastRun.val;
        case '3_3_pattern': return lastRun.len === 3 ? flip(lastRun.val) : lastRun.val;
        case '4_4_pattern': return lastRun.len === 4 ? flip(lastRun.val) : lastRun.val;
        case '1_2_1_pattern':
            if (lastRun.len === 1) return flip(lastRun.val);
            if (lastRun.len === 2) return lastRun.val;
            return null;
        case '2_1_2_pattern':
            if (lastRun.len === 2) return flip(lastRun.val);
            if (lastRun.len === 1) return lastRun.val;
            return null;
        case '3_2_3_pattern':
            if (lastRun.len === 3) return flip(lastRun.val);
            if (lastRun.len === 2) return lastRun.val;
            return null;
        case '2_3_2_pattern':
            if (lastRun.len === 2) return flip(lastRun.val);
            if (lastRun.len === 3) return lastRun.val;
            return null;
        case '4_2_4_pattern':
            if (lastRun.len === 4) return flip(lastRun.val);
            if (lastRun.len === 2) return lastRun.val;
            return null;
        case '1_3_1_pattern':
            if (lastRun.len === 1) return flip(lastRun.val);
            if (lastRun.len === 3) return lastRun.val;
            return null;
        case '3_1_3_pattern':
            if (lastRun.len === 3) return flip(lastRun.val);
            if (lastRun.len === 1) return lastRun.val;
            return null;
        case '1_4_1_pattern':
            if (lastRun.len === 1) return flip(lastRun.val);
            if (lastRun.len === 4) return lastRun.val;
            return null;
        case '2_2_1_pattern':
            if (lastRun.len === 2) return flip(lastRun.val);
            return lastRun.val;
        case '1_1_2_pattern':
            if (lastRun.len === 1) return flip(lastRun.val);
            return lastRun.val;
        case '1_2_2_pattern':
            if (lastRun.len === 1) return flip(lastRun.val);
            return lastRun.val;
        case '2_1_1_pattern':
            if (lastRun.len === 2) return flip(lastRun.val);
            return lastRun.val;
        case '3_3_1_pattern':
            if (lastRun.len === 3) return flip(lastRun.val);
            return lastRun.val;
        case '1_1_3_pattern':
            if (lastRun.len === 1) return flip(lastRun.val);
            return lastRun.val;
        case '1_2_1_2_1_2_pattern':
            return lastRun.len === 1 ? flip(lastRun.val) : lastRun.val;
        case '2_1_2_1_2_1_pattern':
            return lastRun.len === 2 ? flip(lastRun.val) : lastRun.val;
        case '2_2_1_loop_pattern':
            return lastRun.len === 1 ? flip(lastRun.val) : lastRun.val;
        case '1_1_2_loop_pattern':
            return lastRun.len === 2 ? flip(lastRun.val) : lastRun.val;
        case 'fibonacci_pattern':
            return lastRun.len === 1 ? lastRun.val : null;
        case 'reverse_fibonacci_pattern':
            return lastRun.len === 3 ? flip(lastRun.val) : lastRun.val;
        case 'growth_pattern':
            return lastRun.len === 2 ? flip(lastRun.val) : lastRun.val;
        case 'super_long_run_pattern': return flip(lastRun.val);
        case 'long_run_pattern':
            return lastRun.len >= 8 ? flip(lastRun.val) : lastRun.val;
        case 'medium_long_run_pattern': return lastRun.val;
        case 'medium_run_pattern': return lastRun.val;
        case 'short_run_pattern': return flip(lastRun.val);
        case 'broken_2_1_3_2_pattern':
            if (lastRun.len === 2) return flip(lastRun.val);
            if (lastRun.len === 3) return lastRun.val;
            return null;
        case 'broken_3_1_2_3_pattern':
            if (lastRun.len === 3) return flip(lastRun.val);
            return lastRun.val;
        case 'broken_2_3_1_2_pattern':
            if (lastRun.len === 2) return flip(lastRun.val);
            if (lastRun.len === 1) return lastRun.val;
            return null;
        case 'broken_1_3_2_1_pattern':
            if (lastRun.len === 1) return flip(lastRun.val);
            if (lastRun.len === 2) return lastRun.val;
            return null;
        case 'mirror_pattern_12321':
            if (lastRun.len === 1) return flip(lastRun.val);
            if (lastRun.len === 2) return lastRun.val;
            return null;
        case 'alternating_mirror_pattern':
            if (lastRun.len === 2) return flip(lastRun.val);
            if (lastRun.len === 3) return lastRun.val;
            return null;
        case 'staircase_up_pattern':
            if (lastRun.len === 3) return flip(lastRun.val);
            return lastRun.val;
        case 'staircase_down_pattern':
            if (lastRun.len === 1) return flip(lastRun.val);
            return lastRun.val;
        case '2_2_2_1_pattern': return lastRun.len === 1 ? lastRun.val : flip(lastRun.val);
        case '2_2_1_2_pattern': return lastRun.len === 2 ? lastRun.val : flip(lastRun.val);
        case '2_1_2_2_pattern': return lastRun.len === 2 ? lastRun.val : flip(lastRun.val);
        case '1_2_2_2_pattern': return lastRun.len === 2 ? flip(lastRun.val) : lastRun.val;
        default: return null;
    }
}
// =====================================================================
// CORE ALGORITHMS (1-11) — Giữ nguyên từ file gốc
// =====================================================================

function algo5_freqRebalance(history) {
    if (history.length < 20) return null;
    const features = extractFeatures(history);
    const { freq, entropy: e } = features;
    const tCount = freq['T'] || 0;
    const xCount = freq['X'] || 0;
    const diff = Math.abs(tCount - xCount);
    const total = tCount + xCount;
    let threshold = e > 0.9 ? 0.45 : (e < 0.4 ? 0.65 : 0.55);
    const recent = history.slice(-30);
    const recentT = recent.filter(h => h.tx === 'T').length;
    const recentX = recent.filter(h => h.tx === 'X').length;
    const recentTotal = recentT + recentX;
    if (total > 0 && recentTotal > 0) {
        const combinedRatio = (diff / total) * 0.4 + (Math.abs(recentT - recentX) / recentTotal) * 0.6;
        if (combinedRatio > threshold) {
            if (recentT > recentX + 2) return 'X';
            if (recentX > recentT + 2) return 'T';
        }
    }
    return null;
}

function algoA_markov(history) {
    if (history.length < 15) return null;
    const tx = history.map(h => h.tx);
    let maxOrder = 5;
    if (history.length < 30) maxOrder = 3;
    if (history.length < 20) maxOrder = 2;
    let bestPred = null, bestScore = -1;
    for (let order = 2; order <= maxOrder; order++) {
        if (tx.length < order + 8) continue;
        const transitions = {};
        const totalTransitions = tx.length - order;
        const decayFactor = 0.95;
        for (let i = 0; i < totalTransitions; i++) {
            const key = tx.slice(i, i + order).join('');
            const next = tx[i + order];
            const weight = Math.pow(decayFactor, totalTransitions - i - 1);
            if (!transitions[key]) transitions[key] = { T: 0, X: 0 };
            transitions[key][next] += weight;
        }
        const lastKey = tx.slice(-order).join('');
        const counts = transitions[lastKey];
        if (counts && (counts.T + counts.X) > 0.5) {
            const total = counts.T + counts.X;
            const confidence = Math.abs(counts.T - counts.X) / total;
            const pred = counts.T > counts.X ? 'T' : 'X';
            const score = confidence * (order / maxOrder) * Math.min(1, total / 10);
            if (score > bestScore) { bestScore = score; bestPred = pred; }
        }
    }
    return bestPred;
}

function algoB_ngram(history) {
    if (history.length < 30) return null;
    const tx = history.map(h => h.tx);
    const ngramSizes = [7, 6, 5, 4, 3];
    let bestPred = null, bestConfidence = 0;
    for (const n of ngramSizes) {
        if (tx.length < n * 2) continue;
        const target = tx.slice(-n).join('');
        const matches = [];
        for (let i = 0; i <= tx.length - n - 1; i++) {
            if (tx.slice(i, i + n).join('') === target) {
                matches.push({ position: i, next: tx[i + n], distance: tx.length - i });
            }
        }
        if (matches.length >= 2) {
            const weights = { T: 0, X: 0 };
            let totalWeight = 0;
            for (const m of matches) {
                const w = 1 / (m.distance * 0.5 + 1);
                weights[m.next] += w;
                totalWeight += w;
            }
            if (totalWeight > 0) {
                const confidence = Math.abs(weights.T - weights.X) / totalWeight;
                if (confidence > bestConfidence) {
                    bestConfidence = confidence;
                    bestPred = weights.T > weights.X ? 'T' : 'X';
                }
            }
        }
    }
    return bestConfidence > 0.3 ? bestPred : null;
}

function algoS_NeoPattern(history) {
    if (history.length < 20) return null;
    const features = extractFeatures(history);
    const { runs, tx } = features;
    const patternType = detectPatternType(runs);
    if (!patternType || patternType === 'random_pattern') return null;
    const lastTx = tx[tx.length - 1];
    const prediction = predictNextFromPattern(patternType, runs, lastTx);
    if (prediction) {
        const recentRuns = runs.slice(-Math.min(8, runs.length));
        const consistency = recentRuns.length ? recentRuns.filter(r => r.len >= 1).length / recentRuns.length : 0;
        if (consistency > 0.6) return prediction;
    }
    return null;
}

function algoF_SuperDeepAnalysis(history) {
    if (history.length < 40) return null;
    const timeframes = [
        { lookback: 10, weight: 0.3 },
        { lookback: 20, weight: 0.3 },
        { lookback: 40, weight: 0.4 }
    ];
    let totalScore = { T: 0, X: 0 };
    let totalWeight = 0;
    for (const tf of timeframes) {
        if (history.length < tf.lookback) continue;
        const slice = history.slice(-tf.lookback);
        const sliceTx = slice.map(h => h.tx);
        const sliceTotals = slice.map(h => h.total);
        const tCount = sliceTx.filter(t => t === 'T').length;
        const xCount = sliceTx.filter(t => t === 'X').length;
        const meanTotal = avg(sliceTotals);
        const volatility = Math.sqrt(avg(sliceTotals.map(t => Math.pow(t - meanTotal, 2))));
        let tScore = 0, xScore = 0;
        if (meanTotal > 12) xScore += 0.4;
        if (meanTotal < 9) tScore += 0.4;
        if (tCount > xCount + 3) xScore += 0.3;
        if (xCount > tCount + 3) tScore += 0.3;
        if (volatility > 4) {
            if (sliceTx[sliceTx.length - 1] === 'T') tScore += 0.2;
            else xScore += 0.2;
        }
        const trend = sliceTotals[sliceTotals.length - 1] - sliceTotals[0];
        if (trend > 3) xScore += 0.1;
        if (trend < -3) tScore += 0.1;
        const w = tf.weight * (sliceTx.length / tf.lookback);
        totalScore.T += tScore * w;
        totalScore.X += xScore * w;
        totalWeight += w;
    }
    if (totalWeight > 0 && Math.abs(totalScore.T - totalScore.X) > 0.15) {
        return totalScore.T > totalScore.X ? 'T' : 'X';
    }
    return null;
}

function algoE_Transformer(history) {
    if (history.length < 80) return null;
    const tx = history.map(h => h.tx);
    const seqLengths = [5, 6, 8, 10, 12];
    let attentionScores = { T: 0, X: 0 };
    for (const seqLen of seqLengths) {
        if (tx.length < seqLen * 2) continue;
        const targetSeq = tx.slice(-seqLen).join('');
        let seqMatches = 0;
        for (let i = 0; i <= tx.length - seqLen - 1; i++) {
            const historySeq = tx.slice(i, i + seqLen).join('');
            const matchScore = similarity(historySeq, targetSeq);
            if (matchScore >= 0.65) {
                const nextResult = tx[i + seqLen];
                const recency = 1 / (tx.length - i);
                const lengthFactor = seqLen / 12;
                const w = matchScore * recency * lengthFactor;
                attentionScores[nextResult] = (attentionScores[nextResult] || 0) + w;
                seqMatches++;
            }
        }
        if (seqMatches >= 3) {
            const boost = Math.min(1.5, seqMatches / 2);
            attentionScores.T *= boost;
            attentionScores.X *= boost;
        }
    }
    if (attentionScores.T + attentionScores.X > 0.2) {
        const total = attentionScores.T + attentionScores.X;
        const confidence = Math.abs(attentionScores.T - attentionScores.X) / total;
        if (confidence > 0.2) return attentionScores.T > attentionScores.X ? 'T' : 'X';
    }
    return null;
}

function algoG_SuperBridgePredictor(history) {
    const features = extractFeatures(history);
    const { runs } = features;
    if (runs.length < 4) return null;
    const lastRun = runs[runs.length - 1];
    let prediction = null, confidence = 0;

    if (lastRun.len >= 10) { prediction = lastRun.val === 'T' ? 'X' : 'T'; confidence = 0.9; }
    else if (lastRun.len >= 8) { prediction = lastRun.val === 'T' ? 'X' : 'T'; confidence = 0.8; }
    else if (lastRun.len >= 7) {
        const avgRun = avg(runs.map(r => r.len));
        if (lastRun.len > avgRun * 2) { prediction = lastRun.val === 'T' ? 'X' : 'T'; confidence = 0.75; }
        else { prediction = lastRun.val; confidence = 0.6; }
    }
    else if (lastRun.len >= 5 && lastRun.len <= 6) {
        const avgRun = avg(runs.map(r => r.len));
        if (lastRun.len > avgRun * 1.8) { prediction = lastRun.val === 'T' ? 'X' : 'T'; confidence = 0.7; }
        else { prediction = lastRun.val; confidence = 0.62; }
    }

    if (!prediction && runs.length >= 5) {
        const last5Runs = runs.slice(-5);
        const lengths = last5Runs.map(r => r.len);
        if (lengths[0] === 1 && lengths[1] === 1 && lengths[2] >= 3) {
            if (lastRun.len >= 3) { prediction = lastRun.val === 'T' ? 'X' : 'T'; confidence = 0.7; }
        }
        if (lengths.length >= 4) {
            if (lengths[0] === 2 && lengths[1] === 3 && lengths[2] === 2 && lengths[3] === 3) {
                prediction = lastRun.val; confidence = 0.6;
            }
        }
    }

    if (!prediction && runs.length >= 8) {
        const recentRuns = runs.slice(-8);
        const runLengths = recentRuns.map(r => r.len);
        const meanLength = avg(runLengths);
        const stdLength = Math.sqrt(avg(runLengths.map(l => Math.pow(l - meanLength, 2))));
        if (lastRun.len > meanLength + stdLength * 1.5) {
            prediction = lastRun.val === 'T' ? 'X' : 'T';
            confidence = 0.6;
        }
    }
    return confidence > 0.55 ? prediction : null;
}

function algoH_AdaptiveMarkov(history) {
    if (history.length < 25) return null;
    const tx = history.map(h => h.tx);
    let ensembleVotes = { T: 0, X: 0 };
    for (const order of [2, 3, 4, 5]) {
        if (tx.length < order + 5) continue;
        const transitions = {};
        for (let i = 0; i <= tx.length - order - 1; i++) {
            const key = tx.slice(i, i + order).join('');
            const next = tx[i + order];
            if (!transitions[key]) transitions[key] = { T: 0, X: 0 };
            transitions[key][next]++;
        }
        const lastKey = tx.slice(-order).join('');
        const counts = transitions[lastKey];
        if (counts && counts.T + counts.X >= 2) {
            const pred = counts.T > counts.X ? 'T' : 'X';
            const conf = Math.abs(counts.T - counts.X) / (counts.T + counts.X);
            ensembleVotes[pred] += conf * (order / 10);
        }
    }
    for (const lookback of [10, 20, 30]) {
        if (tx.length < lookback) continue;
        const recent = tx.slice(-lookback);
        const tCount = recent.filter(t => t === 'T').length;
        const xCount = recent.filter(t => t === 'X').length;
        if (Math.abs(tCount - xCount) > lookback * 0.2) {
            const pred = tCount > xCount ? 'X' : 'T';
            const conf = Math.abs(tCount - xCount) / lookback;
            ensembleVotes[pred] += conf * 0.5;
        }
    }
    for (const window of [5, 10, 15]) {
        if (tx.length < window * 2) continue;
        const firstHalf = tx.slice(-window * 2, -window);
        const secondHalf = tx.slice(-window);
        const firstT = firstHalf.filter(t => t === 'T').length;
        const secondT = secondHalf.filter(t => t === 'T').length;
        const momentumT = secondT - firstT;
        const momentumX = window - secondT - (window - firstT);
        if (Math.abs(momentumT - momentumX) > window * 0.3) {
            const pred = momentumT > momentumX ? 'T' : 'X';
            const conf = Math.abs(momentumT - momentumX) / window;
            ensembleVotes[pred] += conf * 0.3;
        }
    }
    if (ensembleVotes.T + ensembleVotes.X > 0.3) {
        return ensembleVotes.T > ensembleVotes.X ? 'T' : 'X';
    }
    return null;
}

function algoI_PatternMaster(history) {
    if (history.length < 25) return null;
    const features = extractFeatures(history);
    const { runs, tx } = features;
    if (runs.length < 5) return null;
    const recentRuns = runs.slice(-Math.min(8, runs.length));
    const runLengths = recentRuns.map(r => r.len);
    const runValues = recentRuns.map(r => r.val);
    let patternStrength = { T: 0, X: 0 };
    const runPattern = runLengths.join('');
    const valuePattern = runValues.join('');
    const lastVal = valuePattern[valuePattern.length - 1];

    const patternLibrary = [
        { p: '12121', pred: lastVal === 'T' ? 'X' : 'T', s: 0.7 },
        { p: '21212', pred: lastVal === 'T' ? 'T' : 'X', s: 0.7 },
        { p: '13131', pred: lastVal, s: 0.65 },
        { p: '31313', pred: lastVal === 'T' ? 'X' : 'T', s: 0.65 },
        { p: '14141', pred: lastVal === 'T' ? 'X' : 'T', s: 0.7 },
        { p: '41414', pred: lastVal, s: 0.7 },
        { p: '24242', pred: lastVal === 'T' ? 'X' : 'T', s: 0.65 },
        { p: '42424', pred: lastVal, s: 0.65 },
        { p: '121212', pred: lastVal === 'T' ? 'X' : 'T', s: 0.75 },
        { p: '212121', pred: lastVal === 'T' ? 'T' : 'X', s: 0.75 },
        { p: '221221', pred: lastVal === 'T' ? 'T' : 'X', s: 0.7 },
        { p: '112112', pred: lastVal === 'T' ? 'X' : 'T', s: 0.7 },
        { p: '1231231', pred: lastVal === 'T' ? 'T' : 'X', s: 0.72 },
        { p: '3213213', pred: lastVal === 'T' ? 'X' : 'T', s: 0.72 }
    ];
    for (const lib of patternLibrary) {
        if (runPattern.includes(lib.p)) patternStrength[lib.pred] += lib.s;
    }

    const last10Tx = tx.slice(-10).join('');
    const last12Tx = tx.slice(-12).join('');
    const txPatterns = [
        { p: 'TXTXTXTX', pred: 'X', s: 0.8 },
        { p: 'XTXTXTXT', pred: 'T', s: 0.8 },
        { p: 'TXTXTXTXTX', pred: 'X', s: 0.82 },
        { p: 'XTXTXTXTXT', pred: 'T', s: 0.82 },
        { p: 'TTXXTTXX', pred: 'X', s: 0.7 },
        { p: 'XXTTXXTT', pred: 'T', s: 0.7 },
        { p: 'TTTXXXTT', pred: 'T', s: 0.75 },
        { p: 'XXXTTTXX', pred: 'X', s: 0.75 },
        { p: 'TTXTTXTT', pred: 'X', s: 0.7 },
        { p: 'XXTXXTXX', pred: 'T', s: 0.7 },
        { p: 'TTTXXTTT', pred: 'T', s: 0.72 },
        { p: 'XXXTTXXX', pred: 'X', s: 0.72 },
        { p: 'TXXTXXT', pred: 'X', s: 0.68 },
        { p: 'XTTXTTX', pred: 'T', s: 0.68 }
    ];
    for (const p of txPatterns) {
        if (last10Tx.includes(p.p)) patternStrength[p.pred] += p.s;
        if (last12Tx.includes(p.p)) patternStrength[p.pred] += p.s * 0.3;
    }

    const lastRun = recentRuns[recentRuns.length - 1];
    if (lastRun) {
        const avgRecentLength = avg(runLengths);
        if (lastRun.len > avgRecentLength * 1.8) {
            patternStrength[lastRun.val === 'T' ? 'X' : 'T'] += 0.5;
        } else if (lastRun.len < avgRecentLength * 0.6) {
            patternStrength[lastRun.val] += 0.4;
        }
    }
    if (patternStrength.T + patternStrength.X > 0) {
        const total = patternStrength.T + patternStrength.X;
        const conf = Math.abs(patternStrength.T - patternStrength.X) / total;
        if (conf > 0.25) return patternStrength.T > patternStrength.X ? 'T' : 'X';
    }
    return null;
}

function algoJ_QuantumEntropy(history) {
    if (history.length < 30) return null;
    const features = extractFeatures(history);
    const { entropy: e, tx, runs } = features;
    let entropyPredictions = { T: 0, X: 0 };
    for (const window of [10, 20, 30]) {
        if (tx.length < window) continue;
        const windowTx = tx.slice(-window);
        const windowEntropy = entropy(windowTx);
        if (windowEntropy < 0.3) {
            entropyPredictions[windowTx[windowTx.length - 1]] += 0.6;
        } else if (windowEntropy > 0.9) {
            const tCount = windowTx.filter(t => t === 'T').length;
            const xCount = windowTx.filter(t => t === 'X').length;
            if (tCount > xCount) entropyPredictions['X'] += 0.5;
            else if (xCount > tCount) entropyPredictions['T'] += 0.5;
        } else {
            const recentRuns = runs.slice(-4);
            if (recentRuns.length >= 3) {
                const runLengths = recentRuns.map(r => r.len);
                if (Math.max(...runLengths) - Math.min(...runLengths) <= 2) {
                    entropyPredictions[tx[tx.length - 1]] += 0.4;
                }
            }
        }
    }
    if (e < 0.4) entropyPredictions[tx[tx.length - 1]] += 0.3;
    else if (e > 0.95) {
        const recentT = tx.slice(-20).filter(t => t === 'T').length;
        const recentX = tx.slice(-20).filter(t => t === 'X').length;
        if (recentT > recentX) entropyPredictions['X'] += 0.4;
        else if (recentX > recentT) entropyPredictions['T'] += 0.4;
    }
    if (entropyPredictions.T + entropyPredictions.X > 0.4) {
        return entropyPredictions.T > entropyPredictions.X ? 'T' : 'X';
    }
    return null;
}

function algoK_AntiStuckDetector(history, recentPredictions = []) {
    if (history.length < 15) return null;
    const features = extractFeatures(history);
    const { runs } = features;
    const lastRun = runs[runs.length - 1];
    if (!lastRun) return null;

    const last3Preds = recentPredictions.slice(-3);
    const last3Actual = history.slice(-3).map(h => h.tx);
    if (last3Preds.length === 3) {
        const samePred = last3Preds.every(p => p === last3Preds[0]);
        const allWrong = last3Preds.every((p, i) => p !== last3Actual[i]);
        if (samePred && allWrong) {
            return last3Preds[0] === 'T' ? 'X' : 'T';
        }
    }
    if (lastRun.len >= 6) {
        return lastRun.val === 'T' ? 'X' : 'T';
    }
    return null;
}

function algoL_MomentumReversal(history) {
    if (history.length < 30) return null;
    const totals = history.map(h => h.total);
    const mean5 = avg(totals.slice(-5));
    const mean20 = avg(totals.slice(-20));
    const last3 = totals.slice(-3);
    const last5 = totals.slice(-5);
    const trend3to5 = avg(last3) - avg(last5);
    if (mean5 > 12.5 && trend3to5 > 0.5) return 'X';
    if (mean5 < 8.5 && trend3to5 < -0.5) return 'T';
    if (mean5 > mean20 + 2) return 'X';
    if (mean5 < mean20 - 2) return 'T';
    return null;
}

// =====================================================================
// 4 THUẬT TOÁN MỚI — RÚT RA TỪ INSIGHT 13 FILE .TXT
// =====================================================================

// M: POINT-CENTRIC — Insight: điểm 10/11 là vùng biên, điểm 12 nhiều nhất
function algoM_PointCentric(history) {
    if (history.length < 30) return null;
    const totals = history.map(h => h.total);
    const tx = history.map(h => h.tx);
    const last20 = totals.slice(-20);

    const borderCount10 = last20.filter(t => t === 10).length;
    const borderCount11 = last20.filter(t => t === 11).length;
    const count12 = last20.filter(t => t === 12).length;

    // Insight: điểm 10 nhiều hơn 11 → đang nghiêng X → đảo sang X
    if (borderCount10 >= 4 && borderCount10 > borderCount11 + 2) return 'X';
    if (borderCount11 >= 4 && borderCount11 > borderCount10 + 2) return 'T';

    // Insight: điểm 12 nhiều → Tài nhiều → sắp Xỉu
    if (count12 >= 5) {
        const lastTx = tx[tx.length - 1];
        if (lastTx === 'T') return 'X';
    }

    // Insight: 3 phiên biên liên tiếp → đảo
    const last3 = totals.slice(-3);
    if (last3.every(t => t === 10 || t === 11)) {
        const lastTx = tx[tx.length - 1];
        return lastTx === 'T' ? 'X' : 'T';
    }

    const last10 = totals.slice(-10);
    const mean5 = last10.slice(-5).reduce((a, b) => a + b, 0) / 5;
    const mean10 = last10.reduce((a, b) => a + b, 0) / 10;

    if (mean5 - mean10 > 1.5) return 'X';
    if (mean10 - mean5 > 1.5) return 'T';
    return null;
}

// N: STREAK-LENGTH REGRESSION — Insight: streak TB 1.7-2.2, max 8-9
function algoN_StreakRegression(history) {
    if (history.length < 40) return null;
    const tx = history.map(h => h.tx);
    const runs = [];
    let cur = tx[0], len = 1;
    for (let i = 1; i < tx.length; i++) {
        if (tx[i] === cur) len++;
        else { runs.push({ val: cur, len }); cur = tx[i]; len = 1; }
    }
    runs.push({ val: cur, len });
    if (runs.length < 5) return null;

    const recentRuns = runs.slice(-30);
    const avgStreak = recentRuns.reduce((s, r) => s + r.len, 0) / recentRuns.length;
    const lastRun = runs[runs.length - 1];

    // Insight: streak > 1.5× avg → sắp đảo
    if (lastRun.len > avgStreak * 1.5) return lastRun.val === 'T' ? 'X' : 'T';
    if (lastRun.len < avgStreak * 0.5) return lastRun.val;

    const prevRun = runs[runs.length - 2];
    if (prevRun && prevRun.len > lastRun.len) {
        return lastRun.val === 'T' ? 'X' : 'T';
    }
    return null;
}

// O: CYCLIC PERIOD — Insight: ~40% file có chu kỳ 3-5 phiên
function algoO_CyclicPeriod(history) {
    if (history.length < 30) return null;
    const tx = history.map(h => h.tx);
    let bestCycle = null, bestScore = 0;

    for (let cycle = 3; cycle <= 7; cycle++) {
        if (tx.length < cycle * 3) continue;
        const target = tx.slice(-cycle).join('');
        let matches = 0, total = 0;

        for (let i = tx.length - cycle * 2; i >= 0; i -= cycle) {
            const past = tx.slice(i, i + cycle).join('');
            total++;
            let sim = 0;
            for (let j = 0; j < cycle; j++) if (past[j] === target[j]) sim++;
            if (sim >= cycle - 1) matches++;
        }
        const score = total > 0 ? matches / total : 0;
        if (score > bestScore && total >= 3) {
            bestScore = score;
            bestCycle = cycle;
        }
    }

    if (bestCycle && bestScore >= 0.5) {
        const cyclePattern = tx.slice(-bestCycle);
        return cyclePattern[0];
    }
    return null;
}

// P: BALANCE PRESSURE — Insight: T/X luôn ~50/50 dài hạn
function algoP_BalancePressure(history) {
    if (history.length < 50) return null;
    const tx = history.map(h => h.tx);
    const windows = [20, 50, 100, 200];
    let totalPressure = { T: 0, X: 0 };

    for (const w of windows) {
        if (tx.length < w) continue;
        const slice = tx.slice(-w);
        const t = slice.filter(x => x === 'T').length;
        const x = w - t;
        const imbalance = (t - x) / w;

        if (imbalance > 0.08) totalPressure.X += Math.abs(imbalance) * (w / 100);
        else if (imbalance < -0.08) totalPressure.T += Math.abs(imbalance) * (w / 100);
    }

    if (totalPressure.T > 0.15 && totalPressure.T > totalPressure.X * 1.3) return 'T';
    if (totalPressure.X > 0.15 && totalPressure.X > totalPressure.T * 1.3) return 'X';
    return null;
}

// =====================================================================
// ALGORITHM LIST (15 thuật toán)
// =====================================================================
const ALL_ALGS = [
    { id: 'algo5_freqrebalance', fn: algo5_freqRebalance },
    { id: 'a_markov', fn: algoA_markov },
    { id: 'b_ngram', fn: algoB_ngram },
    { id: 's_neo_pattern', fn: algoS_NeoPattern },
    { id: 'f_super_deep_analysis', fn: algoF_SuperDeepAnalysis },
    { id: 'e_transformer', fn: algoE_Transformer },
    { id: 'g_super_bridge_predictor', fn: algoG_SuperBridgePredictor },
    { id: 'h_adaptive_markov', fn: algoH_AdaptiveMarkov },
    { id: 'i_pattern_master', fn: algoI_PatternMaster },
    { id: 'j_quantum_entropy', fn: algoJ_QuantumEntropy },
    { id: 'l_momentum_reversal', fn: algoL_MomentumReversal },
    { id: 'm_point_centric', fn: algoM_PointCentric },
    { id: 'n_streak_regression', fn: algoN_StreakRegression },
    { id: 'o_cyclic_period', fn: algoO_CyclicPeriod },
    { id: 'p_balance_pressure', fn: algoP_BalancePressure }
];
// =====================================================================
// ENSEMBLE CLASSIFIER (CHỐNG TREO)
// =====================================================================
class SEIUEnsemble {
    constructor(algorithms, opts = {}) {
        this.algs = algorithms;
        this.weights = {};
        this.emaAlpha = opts.emaAlpha ?? 0.08;
        this.minWeight = opts.minWeight ?? 0.02;
        this.historyWindow = opts.historyWindow ?? 700;
        this.performanceHistory = {};
        this.patternMemory = {};
        this.recentPredictions = [];
        this.recentActuals = [];
        this.stuckCounter = 0;
        this.lastPrediction = null;

        for (const a of algorithms) {
            this.weights[a.id] = 1.0;
            this.performanceHistory[a.id] = [];
        }
    }

    fitInitial(history) {
        const window = lastN(history, Math.min(this.historyWindow, history.length));
        if (window.length < 30) return;
        const algScores = {};
        for (const a of this.algs) algScores[a.id] = 0;
        const evalSamples = Math.min(60, window.length - 15);
        const startIdx = window.length - evalSamples;

        for (let i = Math.max(15, startIdx); i < window.length; i++) {
            const prefix = window.slice(0, i);
            const actual = window[i].tx;
            const features = extractFeatures(prefix);
            const patternType = detectPatternType(features.runs);
            for (const a of this.algs) {
                try {
                    const pred = a.fn(prefix);
                    if (pred && pred === actual) {
                        algScores[a.id] += 1;
                        if (patternType) {
                            const key = `${a.id}_${patternType}`;
                            this.patternMemory[key] = (this.patternMemory[key] || 0) + 1;
                        }
                    }
                } catch (e) {}
            }
        }

        let totalWeight = 0;
        for (const id in algScores) {
            const accuracy = algScores[id] / evalSamples;
            this.weights[id] = Math.max(this.minWeight, 0.3 + accuracy * 0.7);
            totalWeight += this.weights[id];
        }
        if (totalWeight > 0) {
            for (const id in this.weights) this.weights[id] /= totalWeight;
        }
        console.log(`⚖️ Khởi tạo trọng số ${Object.keys(this.weights).length} thuật toán.`);
    }

    updateWithOutcome(historyPrefix, actualTx) {
        if (historyPrefix.length < 10) return;
        this.recentActuals.push(actualTx);
        if (this.recentActuals.length > 20) this.recentActuals.shift();

        if (this.lastPrediction) {
            const wasCorrect = this.lastPrediction === actualTx;
            if (wasCorrect) this.stuckCounter = 0;
            else this.stuckCounter++;
        }

        const features = extractFeatures(historyPrefix);
        const patternType = detectPatternType(features.runs);

        for (const a of this.algs) {
            try {
                const pred = a.fn(historyPrefix);
                const correct = pred === actualTx ? 1 : 0;
                this.performanceHistory[a.id].push(correct);
                if (this.performanceHistory[a.id].length > 60) {
                    this.performanceHistory[a.id].shift();
                }

                const recentPerf = lastN(this.performanceHistory[a.id], 25);
                let wa = 0, ws = 0;
                for (let i = 0; i < recentPerf.length; i++) {
                    const w = Math.pow(0.9, recentPerf.length - i - 1);
                    wa += recentPerf[i] * w;
                    ws += w;
                }
                const recentAccuracy = ws > 0 ? wa / ws : 0.5;

                const last5 = this.performanceHistory[a.id].slice(-5);
                const wrongStreak = last5.filter(x => x === 0).length;
                let penaltyMultiplier = 1.0;
                if (wrongStreak >= 4) penaltyMultiplier = 0.6;
                else if (wrongStreak >= 3) penaltyMultiplier = 0.75;

                let patternBonus = 0;
                if (patternType) {
                    const key = `${a.id}_${patternType}`;
                    const ps = this.patternMemory[key] || 0;
                    if (ps > 3) patternBonus = 0.1;
                }
                const targetWeight = Math.min(1, (recentAccuracy + patternBonus + 0.1) * penaltyMultiplier);
                const currentWeight = this.weights[a.id] || this.minWeight;
                const newWeight = this.emaAlpha * targetWeight + (1 - this.emaAlpha) * currentWeight;
                this.weights[a.id] = Math.max(this.minWeight, Math.min(1.5, newWeight));

                if (patternType && correct) {
                    const key = `${a.id}_${patternType}`;
                    this.patternMemory[key] = (this.patternMemory[key] || 0) + 1;
                }
            } catch (e) {
                this.weights[a.id] = Math.max(this.minWeight, (this.weights[a.id] || 1) * 0.9);
            }
        }

        const sumWeights = Object.values(this.weights).reduce((s, w) => s + w, 0);
        if (sumWeights > 0) {
            for (const id in this.weights) this.weights[id] /= sumWeights;
        }
    }

    predict(history) {
        if (history.length < 12) {
            const fallback = 'T';
            this.lastPrediction = fallback;
            this.recentPredictions.push(fallback);
            if (this.recentPredictions.length > 20) this.recentPredictions.shift();
            return { prediction: 'tài', confidence: 0.5, rawPrediction: fallback, algorithmVotes: {} };
        }

        const features = extractFeatures(history);
        const patternType = detectPatternType(features.runs);
        const votes = { T: 0, X: 0 };
        const algorithmDetails = [];
        const algorithmVotes = {};

        for (const a of this.algs) {
            try {
                const pred = a.fn(history);
                if (!pred) continue;
                let weight = this.weights[a.id] || this.minWeight;
                if (patternType) {
                    const key = `${a.id}_${patternType}`;
                    if ((this.patternMemory[key] || 0) > 2) weight *= 1.2;
                }
                votes[pred] = (votes[pred] || 0) + weight;
                algorithmDetails.push({ algorithm: a.id, prediction: pred, weight });
                algorithmVotes[a.id] = pred;
            } catch (e) {}
        }

        try {
            const antiStuckVote = algoK_AntiStuckDetector(history, this.recentPredictions);
            if (antiStuckVote) {
                votes[antiStuckVote] = (votes[antiStuckVote] || 0) + 1.5;
                algorithmDetails.push({ algorithm: 'k_anti_stuck', prediction: antiStuckVote, weight: 1.5 });
                algorithmVotes['k_anti_stuck'] = antiStuckVote;
            }
        } catch (e) {}

        if (votes.T === 0 && votes.X === 0) {
            const fallback = 'T';
            this.lastPrediction = fallback;
            this.recentPredictions.push(fallback);
            if (this.recentPredictions.length > 20) this.recentPredictions.shift();
            return { prediction: 'tài', confidence: 0.5, rawPrediction: fallback, algorithmVotes };
        }

        let { key: best, val: bestVal } = majority(votes);
        const totalVotes = votes.T + votes.X;
        let baseConfidence = bestVal / totalVotes;

        if (this.stuckCounter >= 3) {
            const flipped = best === 'T' ? 'X' : 'T';
            console.log(`🚨 ANTI-STUCK: Đảo ${best} → ${flipped} (stuck ${this.stuckCounter})`);
            best = flipped;
            baseConfidence = Math.max(baseConfidence, 0.6);
            this.stuckCounter = 0;
        }

        const last4Preds = this.recentPredictions.slice(-4);
        if (last4Preds.length === 4 && last4Preds.every(p => p === best)) {
            const last4Actuals = this.recentActuals.slice(-4);
            const wrongCount = last4Preds.filter((p, i) => p !== last4Actuals[i]).length;
            if (wrongCount >= 3) {
                const flipped = best === 'T' ? 'X' : 'T';
                console.log(`🚨 FORCE FLIP: 4 preds giống & sai >=3 → ${best} → ${flipped}`);
                best = flipped;
                baseConfidence = Math.max(baseConfidence, 0.58);
            }
        }

        let consensusBonus = 0;
        const tAlgs = algorithmDetails.filter(a => a.prediction === 'T').length;
        const xAlgs = algorithmDetails.filter(a => a.prediction === 'X').length;
        const totalAlgs = tAlgs + xAlgs;
        if (totalAlgs > 0) {
            const ratio = Math.max(tAlgs, xAlgs) / totalAlgs;
            if (ratio > 0.7) consensusBonus = 0.08;
            if (ratio > 0.8) consensusBonus = 0.12;
        }

        let confidence = Math.min(0.96, Math.max(0.55, baseConfidence + consensusBonus));
        if (this.stuckCounter >= 2) confidence = Math.min(confidence, 0.75);

        this.lastPrediction = best;
        this.recentPredictions.push(best);
        if (this.recentPredictions.length > 20) this.recentPredictions.shift();

        return {
            prediction: best === 'T' ? 'tài' : 'xỉu',
            confidence,
            rawPrediction: best,
            algorithmVotes
        };
    }
}

// =====================================================================
// MANAGER
// =====================================================================
class SEIUManager {
    constructor(opts = {}) {
        this.history = [];
        this.ensemble = new SEIUEnsemble(ALL_ALGS, {
            emaAlpha: opts.emaAlpha ?? 0.08,
            historyWindow: opts.historyWindow ?? 700
        });
        this.currentPrediction = null;
        this.patternHistory = [];
        this.tracker = new PredictionTracker();
    }

    calculateInitialStats() {
        const minStart = 20;
        if (this.history.length < minStart) return;
        const trainSamples = Math.min(80, this.history.length - minStart);
        const startIdx = this.history.length - trainSamples;
        for (let i = Math.max(minStart, startIdx); i < this.history.length; i++) {
            const prefix = this.history.slice(0, i);
            this.ensemble.updateWithOutcome(prefix, this.history[i].tx);
        }
        console.log(`📊 AI huấn luyện ${trainSamples} mẫu.`);
    }

    loadInitial(lines) {
        this.history = lines;
        this.ensemble.fitInitial(this.history);
        this.calculateInitialStats();
        this.currentPrediction = this.getPrediction();
        const nextSession = this.history.at(-1) ? this.history.at(-1).session + 1 : 'N/A';
        if (typeof nextSession === 'number') {
            this.tracker.recordPrediction(
                nextSession,
                this.currentPrediction.rawPrediction,
                this.currentPrediction.confidence,
                this.currentPrediction.algorithmVotes || {}
            );
        }
        console.log(`🔮 Dự đoán phiên ${nextSession}: ${this.currentPrediction.prediction} (${(this.currentPrediction.confidence * 100).toFixed(0)}%)`);
    }

    pushRecord(record) {
        this.history.push(record);
        if (this.history.length > 500) this.history = this.history.slice(-450);

        const tracked = this.tracker.recordActual(record.session, record.tx);
        if (tracked) {
            const icon = tracked.correct ? "✅" : "❌";
            console.log(`📥 ${record.session} → ${record.result} ${icon} (pred: ${tracked.predicted === 'T' ? 'TÀI' : 'XỈU'})`);
        }

        const prefix = this.history.slice(0, -1);
        if (prefix.length >= 10) this.ensemble.updateWithOutcome(prefix, record.tx);
        this.currentPrediction = this.getPrediction();

        const nextSession = record.session + 1;
        this.tracker.recordPrediction(
            nextSession,
            this.currentPrediction.rawPrediction,
            this.currentPrediction.confidence,
            this.currentPrediction.algorithmVotes || {}
        );

        const features = extractFeatures(this.history);
        const patternType = detectPatternType(features.runs);
        if (patternType) {
            this.patternHistory.push(patternType);
            if (this.patternHistory.length > 20) this.patternHistory.shift();
        }
    }

    getPrediction() {
        return this.ensemble.predict(this.history);
    }
}

const seiuManager = new SEIUManager();
// =====================================================================
// API SERVER
// =====================================================================
const app = fastify({ logger: true });
await app.register(cors, { origin: "*" });

async function fetchAndProcessHistory() {
    try {
        const response = await fetch(API_URL);
        const data = await response.json();
        const newHistory = parseLines(data);

        if (newHistory.length === 0) {
            console.log("⚠️ Không có dữ liệu từ API.");
            return;
        }

        const lastSessionInHistory = newHistory.at(-1);

        if (!currentSessionId) {
            seiuManager.loadInitial(newHistory);
            txHistory = newHistory;
            currentSessionId = lastSessionInHistory.session;
            console.log(`✅ Đã tải ${newHistory.length} phiên lịch sử.`);
        } else if (lastSessionInHistory.session > currentSessionId) {
            const newRecords = newHistory.filter(r => r.session > currentSessionId);
            for (const record of newRecords) {
                seiuManager.pushRecord(record);
                txHistory.push(record);
            }
            if (txHistory.length > 350) txHistory = txHistory.slice(-300);
            currentSessionId = lastSessionInHistory.session;
            if (newRecords.length > 0) {
                console.log(`🆕 Cập nhật ${newRecords.length} phiên. Phiên cuối: ${currentSessionId}`);
            }
        }
    } catch (e) {
        console.error("❌ Lỗi fetch:", e.message);
    }
}

// =====================================================================
// API ENDPOINTS
// =====================================================================

// Endpoint chính - kèm ✅/❌ phiên trước
app.get("/api/lc79/txmd5", async () => {
    const lastResult = txHistory.at(-1) || null;
    const currentPrediction = seiuManager.currentPrediction;

    if (!lastResult || !currentPrediction) {
        return {
            id: "@cskhgiabao",
            phien_truoc: null,
            xuc_xac: null,
            ket_qua: "đang chờ...",
            phien_nay: null,
            du_doan: "chưa có",
            do_tin_cay: "0%",
            ket_qua_du_doan: "⏳"
        };
    }

    const tracked = seiuManager.tracker.records.get(lastResult.session);
    const icon = tracked?.correct === true ? "✅" : (tracked?.correct === false ? "❌" : "⏳");

    return {
        id: "@cskhgiabao",
        phien_truoc: lastResult.session,
        xuc_xac: lastResult.dice,
        ket_qua: (lastResult.result || "").toLowerCase(),
        phien_nay: lastResult.session + 1,
        du_doan: currentPrediction.prediction,
        do_tin_cay: `${(currentPrediction.confidence * 100).toFixed(0)}%`,
        ket_qua_du_doan: icon,
        ti_le_dung: `${(seiuManager.tracker.getOverallAccuracy() * 100).toFixed(1)}%`,
        chuoi_gan_nhat: seiuManager.tracker.getRecentResults(20).join(" ")
    };
});

// History với ✅/❌
app.get("/api/taixiumd5/history", async () => {
    if (!txHistory.length) return { message: "không có dữ liệu lịch sử." };
    const reversed = [...txHistory].sort((a, b) => b.session - a.session);

    const history = [];
    for (let i = 0; i < Math.min(200, reversed.length); i++) {
        const item = reversed[i];
        const t = seiuManager.tracker.records.get(item.session);
        let icon = "⏳";
        if (t?.correct === true) icon = "✅";
        else if (t?.correct === false) icon = "❌";

        history.push({
            phien: item.session,
            xuc_xac: item.dice,
            tong: item.total,
            ket_qua: (item.result || "").toLowerCase(),
            du_doan: t?.predicted ? (t.predicted === 'T' ? 'tài' : 'xỉu') : null,
            dung_sai: icon,
            tin_cay: t?.confidence ? `${(t.confidence * 100).toFixed(0)}%` : null
        });
    }

    return {
        id: "@cskhgiabao",
        total: history.length,
        accuracy: `${(seiuManager.tracker.getOverallAccuracy() * 100).toFixed(2)}%`,
        recent_20: seiuManager.tracker.getRecentResults(20).join(" "),
        history
    };
});

// Stats theo thuật toán
app.get("/api/taixiumd5/stats", async () => {
    return {
        id: "@cskhgiabao",
        overall: {
            total: seiuManager.tracker.totalPred,
            correct: seiuManager.tracker.correctPred,
            accuracy: `${(seiuManager.tracker.getOverallAccuracy() * 100).toFixed(2)}%`
        },
        recent_50: seiuManager.tracker.getRecentResults(50).join(""),
        recent_20: seiuManager.tracker.getRecentResults(20).join(" "),
        per_algorithm: seiuManager.tracker.getAlgoStats(),
        total_sessions: txHistory.length,
        total_algorithms: ALL_ALGS.length + 1
    };
});

app.get("/", async () => {
    return {
        status: "ok",
        msg: "AI Tài Xỉu MD5 Pro v6.0 (Insight từ 13 file .txt)",
        version: "6.0",
        algorithms: ALL_ALGS.length + 2,
        pattern_recognition: "35+ mẫu cầu (rút từ 13 file)",
        anti_stuck: true,
        tracker: "✅/❌ realtime",
        endpoints: [
            "/api/lc79/txmd5",
            "/api/taixiumd5/history",
            "/api/taixiumd5/stats"
        ]
    };
});

// =====================================================================
// START
// =====================================================================
const start = async () => {
    await fetchAndProcessHistory();
    clearInterval(fetchInterval);
    fetchInterval = setInterval(fetchAndProcessHistory, 5000);
    console.log(`🔄 Chu kỳ 5 giây.`);

    try {
        await app.listen({ port: PORT, host: "0.0.0.0" });
    } catch (err) {
        const fs = await import("node:fs");
        const logFile = path.join(__dirname, "server-error.log");
        const errorMsg = `\n=== SERVER ERROR ===\nTime: ${new Date().toISOString()}\nError: ${err.message}\nStack: ${err.stack}\n`;
        console.error(errorMsg);
        fs.writeFileSync(logFile, errorMsg, { encoding: "utf8", flag: "a+" });
        process.exit(1);
    }

    let publicIP = "0.0.0.0";
    try {
        const res = await fetch("https://ifconfig.me/ip");
        publicIP = (await res.text()).trim();
    } catch (e) {
        console.error("❌ Lỗi lấy public IP:", e.message);
    }

    console.log("\n🚀 AI Tài Xỉu MD5 Pro v6.0 - Insight từ 13 file .txt!");
    console.log(`   ➜ Local:   http://localhost:${PORT}/`);
    console.log(`   ➜ Network: http://${publicIP}:${PORT}/\n`);
    console.log("📌 API endpoints:");
    console.log(`   ➜ GET /api/lc79/txmd5         → http://${publicIP}:${PORT}/api/lc79/txmd5`);
    console.log(`   ➜ GET /api/taixiumd5/history  → http://${publicIP}:${PORT}/api/taixiumd5/history`);
    console.log(`   ➜ GET /api/taixiumd5/stats    → http://${publicIP}:${PORT}/api/taixiumd5/stats`);
    console.log(`\n🔧 ${ALL_ALGS.length} thuật toán + anti-stuck:`);
    ALL_ALGS.forEach((alg, i) => console.log(`   ${i + 1}. ${alg.id}`));
    console.log(`   ${ALL_ALGS.length + 1}. k_anti_stuck_detector`);
    console.log(`\n📊 Tracker ✅/❌ realtime — KHÔNG cần 13 file .txt khi chạy.`);
};

start();