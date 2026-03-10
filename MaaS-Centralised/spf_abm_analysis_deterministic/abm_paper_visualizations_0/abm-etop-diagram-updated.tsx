import React from 'react';

const ABMETOPDiagram = () => {
  // Colors matching the user's example
  const colors = {
    agentLayer: '#E6F0FF',       // Light blue
    policyIntervention: '#FFF2E6', // Light orange/peach
    systemOutcomes: '#E6F5E6',   // Light green
    optimization: '#F0E6FF',     // Light purple
    agentDots: '#4169E1',        // Royal blue for agent dots
    policyBars: '#FF8C42',       // Orange for policy bars
    outcomeLine: '#4CAF50',      // Green for outcome graph
    optimizationCircle: '#9370DB', // Medium purple for optimization icon
    arrows: '#555555',           // Dark gray for arrows
    text: '#333333'              // Black for text
  };

  return (
    <div className="flex justify-center items-center w-full">
      <svg width="800" height="500" viewBox="0 0 800 500" className="bg-white rounded-lg">
        {/* Agent Layer Box */}
        <rect x="50" y="100" width="200" height="120" rx="8" fill={colors.agentLayer} stroke="#D0D0D0" strokeWidth="1" />
        <text x="150" y="130" textAnchor="middle" fontWeight="bold" fontSize="18" fill={colors.text}>Agent Layer</text>
        <text x="150" y="155" textAnchor="middle" fontSize="14" fill={colors.text}>S: System, P: Policy</text>
        <text x="150" y="175" textAnchor="middle" fontSize="14" fill={colors.text}>A: Agent, E: Environment</text>
        
        {/* Agent dots */}
        <circle cx="100" cy="200" r="6" fill={colors.agentDots} />
        <circle cx="120" cy="200" r="6" fill={colors.agentDots} />
        <circle cx="140" cy="200" r="6" fill={colors.agentDots} />
        <text x="160" y="204" fontSize="14" fill={colors.agentDots}>...</text>

        {/* Policy Intervention Box */}
        <rect x="300" y="100" width="200" height="120" rx="8" fill={colors.policyIntervention} stroke="#D0D0D0" strokeWidth="1" />
        <text x="400" y="130" textAnchor="middle" fontWeight="bold" fontSize="18" fill={colors.text}>Policy Intervention</text>
        <text x="400" y="155" textAnchor="middle" fontSize="14" fill={colors.text}>Fixed Pool Subsidy</text>
        <text x="400" y="175" textAnchor="middle" fontSize="14" fill={colors.text}>Percentage-based Subsidy</text>
        
        {/* Policy bars */}
        <rect x="350" y="190" width="20" height="30" fill={colors.policyBars} />
        <rect x="380" y="190" width="20" height="20" fill={colors.policyBars} />
        <rect x="410" y="190" width="20" height="15" fill={colors.policyBars} opacity="0.8" />
        <rect x="440" y="190" width="20" height="10" fill={colors.policyBars} opacity="0.6" />

        {/* System Outcomes Box */}
        <rect x="550" y="100" width="200" height="120" rx="8" fill={colors.systemOutcomes} stroke="#D0D0D0" strokeWidth="1" />
        <text x="650" y="130" textAnchor="middle" fontWeight="bold" fontSize="18" fill={colors.text}>System Outcomes</text>
        <text x="650" y="155" textAnchor="middle" fontSize="14" fill={colors.text}>Mode Share Equity (Emode)</text>
        <text x="650" y="175" textAnchor="middle" fontSize="14" fill={colors.text}>Travel Time Equity (Etime)</text>
        <text x="650" y="195" textAnchor="middle" fontSize="14" fill={colors.text}>Total System Travel Time (Ttotal)</text>
        
        {/* Outcomes line graph - moved down to avoid overlapping with text */}
        <polyline points="580,210 600,200 620,220 640,195 660,205 680,190 700,215 720,200" 
                 fill="none" stroke={colors.outcomeLine} strokeWidth="2" />

        {/* Optimization Box */}
        <rect x="300" y="300" width="200" height="120" rx="8" fill={colors.optimization} stroke="#D0D0D0" strokeWidth="1" />
        <text x="400" y="330" textAnchor="middle" fontWeight="bold" fontSize="18" fill={colors.text}>Optimisation</text>
        <text x="400" y="355" textAnchor="middle" fontSize="14" fill={colors.text}>Bayesian Process</text>
        <text x="400" y="375" textAnchor="middle" fontSize="14" fill={colors.text}>Policy Update</text>
        
        {/* Optimization circle/target */}
        <circle cx="400" cy="400" r="15" fill="none" stroke={colors.optimizationCircle} strokeWidth="2" />
        <circle cx="400" cy="400" r="8" fill={colors.optimizationCircle} />
        <circle cx="400" cy="400" r="3" fill="white" />

        {/* Arrows connecting the boxes */}
        {/* Agent Layer to Policy Intervention */}
        <line x1="250" y1="160" x2="290" y2="160" stroke={colors.arrows} strokeWidth="2" strokeLinecap="round" />
        <polygon points="290,160 280,155 280,165" fill={colors.arrows} />
        <text x="270" y="145" textAnchor="middle" fontSize="12" fill={colors.arrows}>generates</text>
        
        {/* Policy Intervention to System Outcomes */}
        <line x1="500" y1="160" x2="540" y2="160" stroke={colors.arrows} strokeWidth="2" strokeLinecap="round" />
        <polygon points="540,160 530,155 530,165" fill={colors.arrows} />
        <text x="520" y="145" textAnchor="middle" fontSize="12" fill={colors.arrows}>affects</text>
        
        {/* System Outcomes to Optimization - replaced with curved path */}
        <path d="M650,220 C650,260 600,300 500,320" fill="none" stroke={colors.arrows} strokeWidth="2" strokeLinecap="round" />
        <polygon points="500,320 510,315 507,325" fill={colors.arrows} />
        <text x="600" y="280" textAnchor="middle" fontSize="12" fill={colors.arrows}>evaluates</text>
        
        {/* Optimization to Agent Layer (Feedback Loop) */}
        <line x1="300" y1="360" x2="150" y2="360" stroke={colors.arrows} strokeWidth="2" strokeLinecap="round" />
        <line x1="150" y1="360" x2="150" y2="230" stroke={colors.arrows} strokeWidth="2" strokeLinecap="round" />
        <polygon points="150,230 145,240 155,240" fill={colors.arrows} />
        <text x="150" y="290" textAnchor="middle" fontSize="12" fontStyle="italic" fill={colors.arrows}>Feedback Loop: Optimised Policy Parameters</text>

        {/* Title at the top */}
        <text x="400" y="50" textAnchor="middle" fontWeight="bold" fontSize="24" fill={colors.text}>ABM-ETOP Framework</text>
        <text x="400" y="75" textAnchor="middle" fontSize="14" fill={colors.text}>Agent-Based Model for Equity-Transport Optimization and Policy</text>
      </svg>
    </div>
  );
};

export default ABMETOPDiagram;
