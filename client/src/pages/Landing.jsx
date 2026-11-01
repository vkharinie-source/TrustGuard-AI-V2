import React from 'react';
import { Link } from 'react-router-dom';
import {
    Activity,
    ArrowRight,
    Briefcase,
    FileCheck,
    Globe,
    GraduationCap,
    ImageIcon,
    LayoutDashboard,
    Lock,
    Mail,
    MessageSquare,
    Newspaper,
    PhoneCall,
    Shield,
    Video,
} from '../components/Icons';

const modules = [
    { icon: ImageIcon, label: 'Deepfake image' },
    { icon: Video, label: 'Video analysis' },
    { icon: Globe, label: 'Website security' },
    { icon: PhoneCall, label: 'Call transcripts' },
    { icon: Mail, label: 'Email security' },
    { icon: MessageSquare, label: 'Messages' },
    { icon: FileCheck, label: 'Applications' },
    { icon: GraduationCap, label: 'Internships' },
    { icon: Briefcase, label: 'Job offers' },
    { icon: Newspaper, label: 'News review' },
];

const steps = [
    { number: '01', title: 'Bring the signal', text: 'Paste text, enter a URL, or provide an image or transcript.' },
    { number: '02', title: 'Review the analysis', text: 'See the detector result, confidence, and any findings returned.' },
    { number: '03', title: 'Choose what to do', text: 'Save a report, listen to the result, or verify through official sources.' },
];

export default function Landing({ onboardingComplete }) {
    const startPath = onboardingComplete ? '/login' : '/onboarding';

    return (
        <div className="public-site">
            <header className="public-nav">
                <Link to="/" className="public-brand" aria-label="TrustGuard AI home">
                    <span className="public-brand-mark"><Shield size={20} /></span>
                    <span>TRUSTGUARD <b>AI</b><small>DIGITAL SECURITY</small></span>
                </Link>
                <nav className="public-nav-links" aria-label="Main navigation">
                    <a href="#capabilities">Capabilities</a>
                    <a href="#workflow">Workflow</a>
                </nav>
                <div className="public-nav-actions">
                    <Link to="/login" className="public-signin">Sign in</Link>
                    <Link to="/register" className="public-nav-cta">Create account <ArrowRight size={15} /></Link>
                </div>
            </header>

            <main>
                <section className="public-hero">
                    <div className="public-hero-copy">
                        <div className="public-eyebrow"><span /> DIGITAL TRUST, ONE WORKSPACE</div>
                        <h1>TrustGuard AI</h1>
                        <h2>Make a more informed call before you click, share, or respond.</h2>
                        <p>
                            Review suspicious images, links, messages, and documents with focused security checks and clear reports.
                            Keep the evidence in view and decide your next step with care.
                        </p>
                        <div className="public-hero-actions">
                            <Link to={startPath} className="public-primary">{onboardingComplete ? 'Sign in to continue' : 'Get started'} <ArrowRight size={17} /></Link>
                            <Link to="/register" className="public-secondary">Create an account</Link>
                        </div>
                        <div className="public-proofline"><Lock size={14} /> Private workspace <span /> Ten analysis areas <span /> Downloadable reports</div>
                    </div>

                    <div className="workspace-preview" aria-label="TrustGuard analysis workspace preview">
                        <div className="workspace-preview-head">
                            <div>
                                <span className="preview-kicker">TRUSTGUARD WORKSPACE</span>
                                <h3>Choose a check</h3>
                            </div>
                            <div className="preview-shield"><Shield size={19} /></div>
                        </div>
                        <div className="preview-rule" />
                        <div className="preview-module-list">
                            {modules.slice(0, 6).map(({ icon: Icon, label }, index) => (
                                <div className="preview-module" key={label}>
                                    <span className="preview-module-icon"><Icon size={17} /></span>
                                    <span>{label}</span>
                                    <span className="preview-index">0{index + 1}</span>
                                </div>
                            ))}
                        </div>
                        <div className="preview-foot"><Activity size={15} /> ANALYSIS STARTS WITH YOUR INPUT</div>
                    </div>
                </section>

                <section className="public-capabilities" id="capabilities">
                    <div className="section-heading">
                        <span className="public-eyebrow">ONE SECURITY WORKSPACE</span>
                        <h2>Different signals. One clear place to inspect them.</h2>
                    </div>
                    <div className="capability-grid">
                        {modules.map(({ icon: Icon, label }) => (
                            <div className="capability-item" key={label}>
                                <Icon size={18} />
                                <span>{label}</span>
                            </div>
                        ))}
                        <div className="capability-item"><LayoutDashboard size={18} /><span>Dashboard and history</span></div>
                    </div>
                </section>

                <section className="public-workflow" id="workflow">
                    <div className="section-heading">
                        <span className="public-eyebrow">A PRACTICAL WORKFLOW</span>
                        <h2>From first concern to considered next step.</h2>
                    </div>
                    <div className="workflow-grid">
                        {steps.map((step) => (
                            <article className="workflow-step" key={step.number}>
                                <span className="workflow-number">{step.number}</span>
                                <h3>{step.title}</h3>
                                <p>{step.text}</p>
                            </article>
                        ))}
                    </div>
                </section>
            </main>

            <footer className="public-footer">
                <span>TRUSTGUARD AI <span className="footer-divider">/</span> DIGITAL SECURITY</span>
                <div><Link to="/login">Sign in</Link><Link to="/register">Create account</Link></div>
            </footer>
        </div>
    );
}
