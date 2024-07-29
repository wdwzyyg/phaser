const path = require('path');
const CopyPlugin = require("copy-webpack-plugin");

module.exports = {
  mode: "development",
  devtool: 'eval-source-map',
  entry: {
    manager: './src/manager.ts',
    dashboard: './src/dashboard.ts',
  },
  output: {
    filename: "bundle-[name].js",
    path: path.resolve(__dirname, "dist"),
  },
  module: {
    rules: [
        {
            test: /\.ts$/,
            use: 'ts-loader',
            include: [path.resolve(__dirname, 'src')],
        },
    ],
  },
  resolve: {
    extensions: ['.ts', '.js'],
  },
  plugins: [
    new CopyPlugin({
      patterns: [
        { from: "src/*.css", to: "[name][ext]" },
      ]
    })
  ],
  experiments: {
    asyncWebAssembly: true
  },
};
