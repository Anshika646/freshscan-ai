import React from 'react';

export type ActiveTab = 'overview' | 'scanner' | 'analytics' | 'dataset' | 'deploy';

interface NavbarProps {
  activeTab: ActiveTab;
  onSelectTab: (tab: ActiveTab) => void;
  onTriggerScanCta: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  onSelectTab,
  onTriggerScanCta,
}) => {
  const navItems: Array<{ id: ActiveTab; label: string }> = [
    { id: 'overview', label: 'Overview' },
    { id: 'scanner', label: 'Scanner' },
    { id: 'analytics', label: 'Analytics' },
    { id: 'dataset', label: 'Model & Data' },
    { id: 'deploy', label: 'Deploy' },
  ];

  return (
    <header className="sticky top-0 z-40 bg-[#070A08]/95 border-b border-[#18221D]">
      <div className="max-w-[1320px] mx-auto flex items-center justify-between gap-8 px-4 sm:px-6 py-3.5">
        {/* Zone 1: Single text element wordmark */}
        <button
          type="button"
          onClick={() => onSelectTab('overview')}
          className="font-display text-lg font-bold tracking-tight text-[#F3F7F4] hover:text-[#34D399] transition-colors whitespace-nowrap shrink-0 cursor-pointer focus-visible:outline-2 focus-visible:outline-[#10B981]"
        >
          FreshScan AI
        </button>

        {/* Zone 2: 5 concise single-line text navigation links */}
        <nav aria-label="Primary" className="flex items-center gap-5 sm:gap-7 overflow-x-auto">
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelectTab(item.id)}
                className={`py-1 text-sm font-medium transition-colors whitespace-nowrap shrink-0 border-b-2 cursor-pointer focus-visible:outline-2 focus-visible:outline-[#10B981] ${
                  isActive
                    ? 'border-[#10B981] text-[#F3F7F4] font-semibold'
                    : 'border-transparent text-[#8FA396] hover:text-[#F3F7F4] hover:border-[#2A3B32]'
                }`}
              >
                {item.label}
              </button>
            );
          })}
        </nav>

        {/* Zone 3: 1 primary action */}
        <div className="flex items-center shrink-0">
          <button
            type="button"
            onClick={onTriggerScanCta}
            className="px-4 py-2 text-xs font-semibold text-white bg-[#059669] rounded-lg hover:bg-[#10B981] transition-colors whitespace-nowrap shrink-0 cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#10B981]"
          >
            Scan Your Produce
          </button>
        </div>
      </div>
    </header>
  );
};
