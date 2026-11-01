import React from 'react';
import { Briefcase } from '../components/Icons';
import AnalysisPage from '../components/AnalysisPage';
import { api } from '../services/api';

export default function Job() {
  return (
    <AnalysisPage
      icon={<Briefcase size={22} color="#818CF8" />}
      title="Job Offer Verifier"
      subtitle="Verify corporate recruitment and flag advance check fraud"
      reportTitle="Job Security Report"
      moduleType="job"
      fields={[
        { key: 'title', label: 'Job Title', placeholder: 'e.g. Senior Software Engineer' },
        { key: 'company', label: 'Company', placeholder: 'e.g. TechCorp Inc.' },
        { key: 'description', label: 'Offer Details', placeholder: 'Paste job offer details, salary, contact info, application link...', multiline: true, rows: 6 },
      ]}
      onSubmit={async (vals) => api.analyzeJob(vals.title, vals.company, vals.description)}
      buildPDFData={(result, vals) => ({
        inputInfo: { 'Job Title': vals.title, Company: vals.company },
        findings: [
          `Verdict: ${String(result.prediction || result.result || '').toUpperCase()}`,
          result.confidence !== undefined ? `Confidence: ${Number(result.confidence).toFixed(2)}%` : null,
        ].filter(Boolean),
        recommendations: [
          'Legitimate employers never ask for payment during recruitment.',
          'Verify the company registration and HR department independently.',
          'Be cautious of unrealistic salary offers and vague job descriptions.',
        ],
        technicalInfo: { 'Analysis Engine': 'Recruitment Fraud Detector', 'Job Title': vals.title, Company: vals.company },
      })}
    />
  );
}
