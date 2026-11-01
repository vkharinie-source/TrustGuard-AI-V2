import React from 'react';
import { GraduationCap } from '../components/Icons';
import AnalysisPage from '../components/AnalysisPage';
import { api } from '../services/api';

export default function Internship() {
  return (
    <AnalysisPage
      icon={<GraduationCap size={22} color="#818CF8" />}
      title="Internship Verifier"
      subtitle="Verify internship offers and detect registration fee scams"
      reportTitle="Internship Security Report"
      moduleType="internship"
      fields={[
        { key: 'title', label: 'Internship Role / Title', placeholder: 'e.g. Data Science Intern' },
        { key: 'company', label: 'Company / Organization', placeholder: 'e.g. GlobalTech Solutions' },
        { key: 'description', label: 'Offer Details / Description', placeholder: 'Paste the internship offer details, contact info, application link, fees, etc.', multiline: true, rows: 6 },
      ]}
      onSubmit={async (vals) => api.analyzeInternship(vals.title, vals.company, vals.description)}
      buildPDFData={(result, vals) => ({
        inputInfo: { Role: vals.title, Company: vals.company },
        findings: [
          `Verdict: ${String(result.prediction || result.result || '').toUpperCase()}`,
          result.confidence !== undefined ? `Confidence: ${Number(result.confidence).toFixed(2)}%` : null,
        ].filter(Boolean),
        recommendations: [
          'Legitimate internships never require upfront registration fees.',
          'Verify the company through official channels and LinkedIn.',
          'Check the company domain for legitimacy before applying.',
        ],
        technicalInfo: { 'Analysis Engine': 'Recruitment Fraud Detector', Role: vals.title, Company: vals.company },
      })}
    />
  );
}
