"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Navigation() {
  const pathname = usePathname();
  const navItems = [
    { href: "/", label: "Workspace" },
    { href: "/routes", label: "Routes" },
  ];

  return (
    <nav className="fixed left-0 right-0 top-0 z-50 border-b border-[#262627] bg-[#0e0e0f]/95 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
        <div className="flex items-center gap-8">
          <Link href="/" className="text-xl font-bold tracking-tighter text-[#cc97ff]">
            RouteNote
          </Link>
          <div className="hidden items-center gap-6 md:flex">
            {navItems.map((item) => {
              const isActive =
                pathname === item.href || (item.href !== "/" && pathname?.startsWith(item.href));
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`pb-1 text-sm font-semibold tracking-tight transition-colors duration-200 ${
                    isActive
                      ? "border-b-2 border-[#cc97ff] text-[#cc97ff]"
                      : "text-[#adaaab] hover:text-[#cc97ff]"
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            className="rounded-md p-2 text-[#adaaab] transition-colors hover:text-[#cc97ff]"
            aria-label="Notifications"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 17h5l-1.4-1.4A2 2 0 0118 14.2V11a6 6 0 10-12 0v3.2a2 2 0 01-.6 1.4L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
            </svg>
          </button>
          <button
            type="button"
            className="rounded-md p-2 text-[#adaaab] transition-colors hover:text-[#cc97ff]"
            aria-label="Settings"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M11.3 2.8l.3 1.8a7.8 7.8 0 012 .8l1.6-1a1 1 0 011.3.2l1.4 1.4a1 1 0 01.2 1.3l-1 1.6c.4.6.7 1.3.8 2l1.8.3a1 1 0 01.8 1v2a1 1 0 01-.8 1l-1.8.3a7.8 7.8 0 01-.8 2l1 1.6a1 1 0 01-.2 1.3l-1.4 1.4a1 1 0 01-1.3.2l-1.6-1a7.8 7.8 0 01-2 .8l-.3 1.8a1 1 0 01-1 .8h-2a1 1 0 01-1-.8l-.3-1.8a7.8 7.8 0 01-2-.8l-1.6 1a1 1 0 01-1.3-.2L2.8 18a1 1 0 01-.2-1.3l1-1.6a7.8 7.8 0 01-.8-2L1 12.8a1 1 0 01-.8-1v-2a1 1 0 01.8-1l1.8-.3a7.8 7.8 0 01.8-2l-1-1.6A1 1 0 012.8 3l1.4-1.4a1 1 0 011.3-.2l1.6 1a7.8 7.8 0 012-.8l.3-1.8a1 1 0 011-.8h2a1 1 0 011 .8zM12 15.5A3.5 3.5 0 1012 8a3.5 3.5 0 000 7.5z" />
            </svg>
          </button>
          <div className="ml-1 h-8 w-8 rounded-full bg-[#262627]" />
        </div>
      </div>
    </nav>
  );
}
