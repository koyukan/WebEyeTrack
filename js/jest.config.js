module.exports = {
    preset: 'ts-jest',
    transform: {
      '^.+\\.(ts|tsx)?$': ['ts-jest', {diagnostics: {ignoreCodes: ['TS151001']}}],
      "^.+\\.(js|jsx)$": "babel-jest",
    }
  };