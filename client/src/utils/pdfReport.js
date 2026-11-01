// ====================================================
// TrustGuard AI — PDF Report Generator
// Uses jsPDF for client-side PDF generation
// ====================================================
import { jsPDF } from 'jspdf';

const COLORS = {
  bg: [13, 19, 34],
  cardBg: [19, 27, 46],
  accent: [99, 102, 241],
  accentLight: [129, 140, 248],
  textPrimary: [249, 250, 251],
  textSecondary: [229, 231, 235],
  textMuted: [156, 163, 175],
  success: [52, 211, 153],
  warning: [251, 191, 36],
  danger: [248, 113, 113],
  border: [31, 41, 55],
};

const getResultColor = (result) => {
  const r = String(result).toUpperCase();
  if (['REAL', 'SAFE', 'VERIFIED'].includes(r)) return COLORS.success;
  if (['SUSPICIOUS', 'PARTIALLY VERIFIED', 'UNVERIFIED'].includes(r)) return COLORS.warning;
  return COLORS.danger;
};

export function generatePDFReport({
  title = 'Analysis Report',
  moduleType = 'General',
  result = {},
  inputInfo = {},
  findings = [],
  details = [],
  recommendations = [],
  technicalInfo = {},
  extraSections = [],
}) {
  const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
  const pageW = doc.internal.pageSize.getWidth();
  const pageH = doc.internal.pageSize.getHeight();
  const margin = 15;
  const contentW = pageW - margin * 2;
  let y = margin;

  const addPage = () => {
    doc.addPage();
    y = margin;
  };

  const checkPageBreak = (needed = 20) => {
    if (y + needed > pageH - 20) addPage();
  };

  // ---- Header Band ----
  doc.setFillColor(...COLORS.bg);
  doc.rect(0, 0, pageW, 45, 'F');
  doc.setFillColor(...COLORS.accent);
  doc.rect(0, 44, pageW, 1.5, 'F');

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(9);
  doc.setTextColor(...COLORS.accentLight);
  doc.text('TRUSTGUARD AI', margin, 14);

  doc.setFontSize(6.5);
  doc.setTextColor(...COLORS.textMuted);
  doc.text('AI-POWERED DIGITAL SECURITY PLATFORM', margin, 19);

  doc.setFontSize(14);
  doc.setTextColor(...COLORS.textPrimary);
  doc.text(title.toUpperCase(), margin, 32);

  doc.setFontSize(7);
  doc.setTextColor(...COLORS.textMuted);
  doc.text(`Generated: ${new Date().toLocaleString()}`, margin, 39);
  doc.text(`Module: ${moduleType}`, pageW - margin - doc.getTextWidth(`Module: ${moduleType}`), 39);

  y = 52;

  // ---- Result Box ----
  const pred = String(result.prediction || result.result || 'N/A').toUpperCase();
  const conf = result.confidence !== undefined ? `${Number(result.confidence).toFixed(2)}%` : 'N/A';
  const risk = result.risk_score !== undefined ? String(result.risk_score) : null;
  const resColor = getResultColor(pred);

  doc.setFillColor(...COLORS.cardBg);
  doc.roundedRect(margin, y, contentW, risk ? 28 : 22, 3, 3, 'F');

  doc.setFontSize(7);
  doc.setTextColor(...COLORS.textMuted);
  doc.text('DETECTION RESULT', margin + 5, y + 6);
  doc.text('CONFIDENCE', margin + 60, y + 6);
  if (risk) doc.text('RISK SCORE', margin + 110, y + 6);

  doc.setFontSize(13);
  doc.setTextColor(...resColor);
  doc.setFont('helvetica', 'bold');
  doc.text(pred, margin + 5, y + 15);

  doc.setFontSize(11);
  doc.setTextColor(...COLORS.textSecondary);
  doc.text(conf, margin + 60, y + 15);

  if (risk) {
    doc.text(risk, margin + 110, y + 15);
  }

  y += risk ? 34 : 28;

  // ---- Input Information ----
  const inputEntries = Object.entries(inputInfo).filter(([, v]) => v);
  if (inputEntries.length > 0) {
    checkPageBreak(30);
    doc.setFontSize(8);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...COLORS.accentLight);
    doc.text('INPUT INFORMATION', margin, y);
    y += 5;

    doc.setFillColor(...COLORS.cardBg);
    const inputH = inputEntries.length * 7 + 4;
    doc.roundedRect(margin, y, contentW, inputH, 2, 2, 'F');
    y += 5;

    inputEntries.forEach(([key, val]) => {
      doc.setFontSize(7);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(...COLORS.textMuted);
      doc.text(`${key}:`, margin + 4, y);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(...COLORS.textSecondary);
      const valText = String(val).substring(0, 90);
      doc.text(valText, margin + 40, y);
      y += 7;
    });
    y += 4;
  }

  // ---- Key Findings ----
  if (findings.length > 0) {
    checkPageBreak(20);
    doc.setFontSize(8);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...COLORS.accentLight);
    doc.text('KEY FINDINGS', margin, y);
    y += 5;

    findings.forEach((f) => {
      checkPageBreak(10);
      doc.setFillColor(COLORS.cardBg[0], COLORS.cardBg[1], COLORS.cardBg[2]);
      doc.roundedRect(margin, y, contentW, 8, 1.5, 1.5, 'F');
      doc.setFontSize(7);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(...COLORS.textSecondary);
      const findingText = String(f).substring(0, 120);
      doc.text(`  ${findingText}`, margin + 3, y + 5.5);
      y += 10;
    });
    y += 3;
  }

  // ---- Detailed Analysis ----
  if (details.length > 0) {
    checkPageBreak(20);
    doc.setFontSize(8);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...COLORS.accentLight);
    doc.text('DETAILED ANALYSIS', margin, y);
    y += 5;

    details.forEach((d) => {
      checkPageBreak(12);
      doc.setFontSize(7);
      if (typeof d === 'object' && d.label) {
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(...COLORS.textSecondary);
        doc.text(d.label, margin + 2, y);
        doc.setFont('helvetica', 'normal');
        doc.setTextColor(...COLORS.textMuted);
        const statusCol = d.flagged ? COLORS.danger : COLORS.success;
        doc.setTextColor(...statusCol);
        doc.text(d.flagged ? 'FLAGGED' : 'CLEAR', margin + 80, y);
        y += 6;
      } else {
        doc.setFont('helvetica', 'normal');
        doc.setTextColor(...COLORS.textSecondary);
        const lines = doc.splitTextToSize(String(d), contentW - 6);
        lines.forEach((line) => {
          checkPageBreak(6);
          doc.text(line, margin + 2, y);
          y += 5;
        });
        y += 2;
      }
    });
    y += 3;
  }

  // ---- Recommendations ----
  if (recommendations.length > 0) {
    checkPageBreak(20);
    doc.setFontSize(8);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...COLORS.accentLight);
    doc.text('RECOMMENDATIONS', margin, y);
    y += 5;

    recommendations.forEach((rec, idx) => {
      checkPageBreak(10);
      doc.setFontSize(7);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(...COLORS.textSecondary);
      const recLines = doc.splitTextToSize(`${idx + 1}. ${String(rec)}`, contentW - 10);
      recLines.forEach((line) => {
        checkPageBreak(6);
        doc.text(line, margin + 3, y);
        y += 5;
      });
      y += 2;
    });
    y += 3;
  }

  // ---- Technical Info ----
  const techEntries = Object.entries(technicalInfo).filter(([, v]) => v !== undefined && v !== null);
  if (techEntries.length > 0) {
    checkPageBreak(20);
    doc.setFontSize(8);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...COLORS.accentLight);
    doc.text('TECHNICAL INFORMATION', margin, y);
    y += 5;

    techEntries.forEach(([key, val]) => {
      checkPageBreak(7);
      doc.setFontSize(7);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(...COLORS.textMuted);
      doc.text(`${key}:`, margin + 3, y);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(...COLORS.textSecondary);
      doc.text(String(val).substring(0, 100), margin + 50, y);
      y += 6;
    });
    y += 3;
  }

  // ---- Extra Sections ----
  extraSections.forEach((section) => {
    checkPageBreak(20);
    doc.setFontSize(8);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...COLORS.accentLight);
    doc.text(String(section.title).toUpperCase(), margin, y);
    y += 5;

    if (typeof section.content === 'string') {
      doc.setFontSize(7);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(...COLORS.textSecondary);
      const lines = doc.splitTextToSize(section.content, contentW - 6);
      lines.forEach((line) => {
        checkPageBreak(6);
        doc.text(line, margin + 2, y);
        y += 5;
      });
    } else if (Array.isArray(section.content)) {
      section.content.forEach((item) => {
        checkPageBreak(7);
        doc.setFontSize(7);
        doc.setFont('helvetica', 'normal');
        doc.setTextColor(...COLORS.textSecondary);
        doc.text(`- ${String(item).substring(0, 120)}`, margin + 3, y);
        y += 6;
      });
    }
    y += 4;
  });

  // ---- Footer ----
  const totalPages = doc.internal.getNumberOfPages();
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i);
    doc.setFillColor(...COLORS.border);
    doc.rect(0, pageH - 12, pageW, 12, 'F');
    doc.setFontSize(6);
    doc.setTextColor(...COLORS.textMuted);
    doc.text('TRUSTGUARD AI - CONFIDENTIAL SECURITY REPORT', margin, pageH - 5);
    doc.text(`Page ${i} of ${totalPages}`, pageW - margin - 20, pageH - 5);
  }

  const safeTitle = title.replace(/[^a-zA-Z0-9]/g, '_').toLowerCase();
  doc.save(`TrustGuard_AI_${safeTitle}_${Date.now()}.pdf`);
}
