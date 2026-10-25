import React from 'react';

export type ActiveTab = 'scanner' | 'analytics' | 'dataset' | 'deploy';

interface NavbarProps {
  activeTab: ActiveTab;
  onSelectTab: (tab: ActiveTab) => void;
  onTriggerUpload: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  onSelectTab,
  onTriggerUpload,
}) => {
  const navItems: Array<{ id: ActiveTab; label: string }> = [
    { id: 'scanner', label: 'Scanner' },
    { id: 'analytics', label: 'Analytics' },
    { id: 'dataset', label: 'Dataset' },
    { id: 'deploy', label: 'GitHub & Deploy' },
  ];

  return (
    <header className="sticky top-0 z-30 bg-[#F7F8F5]/95 backdrop-blur-sm border-b border-[#DFE5DD]">
      <div className="max-w-[1320px] mx-auto flex items-center justify-between gap-8 px-6 py-3.5">
        <button
          type="button"
          onClick={() => onSelectTab('scanner')}
          className="text-lg font-bold tracking-tight text-[#14241B] hover:text-[#1B4332] transition-colors whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-[#1B4332]"
        >
          FreshScan AI
        </button>

        <nav aria-label="Primary" className="flex items-center gap-6 overflow-x-auto">
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelectTab(item.id)}
                className={`py-1 text-sm font-medium transition-colors whitespace-nowrap shrink-0 border-b-2 focus-visible:outline-2 focus-visible:outline-[#1B4332] ${
                  isActive
                    ? 'border-[#1B4332] text-[#14241B] font-semibold'
                    : 'border-transparent text-[#4A5D52] hover:text-[#14241B] hover:border-[#B7C4BC]'
                }`}
              >
                {item.label}
              </button>
            );
          })}
        </nav>

        <div className="flex items-center shrink-0">
          <button
            type="button"
            onClick={onTriggerUpload}
            className="px-4 py-2 text-xs font-semibold text-white bg-[#1B4332] rounded-lg hover:bg-[#143326] transition-colors whitespace-nowrap shrink-0 cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#1B4332]"
          >
            Scan Fruit Image
          </button>
        </div>
      </div>
    </header>
  );
};
