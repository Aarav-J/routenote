import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Navigation from "./components/Navigation";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "RouteNote | Climbing Hold Intelligence",
  description: "Analyze climbing walls, cluster holds by color, and capture route notes with ease.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="bg-[var(--background)]">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased text-[var(--foreground)] bg-transparent`}
      >
        <Navigation />
        {children}
      </body>
    </html>
  );
}
