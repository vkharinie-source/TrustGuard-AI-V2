import React from 'react';
import { FileCheck } from '../components/Icons';
import AnalysisPage from '../components/AnalysisPage';
import { api } from '../services/api';

export default function Application() {
  return (
    <AnalysisPage
      icon={<FileCheck size={22} color="#818CF8" />}
      title="Application Auditor"
      subtitle="Audit mobile app packages for malicious permission risks"
      reportTitle="Application Security Report"
      moduleType="application"
      fields={[
        { key: 'appName', label: 'Application Name', placeholder: 'e.g. SuperClean Pro' },
        { key: 'developer', label: 'Developer', placeholder: 'e.g. Unknown Dev Studio' },
        { key: 'description', label: 'App Description / Details', placeholder: 'Paste any app details, permissions, or package info...', multiline: true, rows: 5 },
      ]}
      onSubmit={async (vals) => api.analyzeApplication(vals.appName, vals.developer, vals.description)}
      buildPDFData={(result, vals) => ({
        inputInfo: { Application: vals.appName, Developer: vals.developer },
        findings: [
          `Verdict: ${String(result.prediction || result.result || '').toUpperCase()}`,
          result.confidence !== undefined ? `Confidence: ${Number(result.confidence).toFixed(2)}%` : null,
        ].filter(Boolean),
        recommendations: [
          'Only install applications from trusted sources like official app stores.',
          'Review requested permissions carefully before installation.',
          'Check developer credibility and user reviews.',
        ],
        technicalInfo: { 'Analysis Engine': 'Package Metadata Inspector', Application: vals.appName, Developer: vals.developer },
      })}
    />
  );
}
