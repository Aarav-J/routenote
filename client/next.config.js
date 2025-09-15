/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable webpack's built-in file watching
  webpack: (config, { dev, isServer }) => {
    if (dev && !isServer) {
      // Optimize file watching for better hot reload performance
      config.watchOptions = {
        poll: 1000, // Check for changes every second
        aggregateTimeout: 300, // Delay before rebuilding
        ignored: /node_modules/,
      };
    }
    return config;
  },
  
  // Experimental features for better development experience
  experimental: {
    // Enable faster refresh
    optimizeCss: false,
  },
  
  // Development configuration
  ...(process.env.NODE_ENV === 'development' && {
    // Disable webpack cache in development if issues persist
    webpack: (config, { dev, isServer }) => {
      if (dev && !isServer) {
        config.cache = false;
        config.watchOptions = {
          poll: 1000,
          aggregateTimeout: 300,
          ignored: /node_modules/,
        };
      }
      return config;
    }
  })
};

module.exports = nextConfig;