#!/usr/bin/env node

/**
 * Copy webeyetrack worker to public folder
 * This ensures the latest worker is always available
 */

const fs = require('fs');
const path = require('path');

const source = path.join(__dirname, '../node_modules/webeyetrack/dist/webeyetrack.worker.js');
const destination = path.join(__dirname, '../public/webeyetrack.worker.js');

try {
  // Check if source exists
  if (!fs.existsSync(source)) {
    console.error('❌ Worker source not found:', source);
    console.error('   Run "npm install" first');
    process.exit(1);
  }

  // Create public directory if it doesn't exist
  const publicDir = path.dirname(destination);
  if (!fs.existsSync(publicDir)) {
    fs.mkdirSync(publicDir, { recursive: true });
  }

  // Copy file
  fs.copyFileSync(source, destination);

  console.log('✅ Copied webeyetrack.worker.js to public/');
} catch (error) {
  console.error('❌ Failed to copy worker:', error.message);
  process.exit(1);
}
