import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from "path"

const repo = process.env.GITHUB_REPOSITORY?.split("/")?.[1];
const isUserPage = repo?.endsWith(".github.io");
const base = isUserPage ? "/" : `/${repo ?? "WebEyeTrack"}/`;

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  base,
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
})
