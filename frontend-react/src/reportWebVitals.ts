// Stubbed web vitals reporter to avoid external dependency during build.
// If you want web vitals later, add `web-vitals` to package.json and restore CRA template.

export type ReportHandler = (metric?: unknown) => void;

const reportWebVitals = (_onPerfEntry?: ReportHandler): void => {
  // no-op
};

export default reportWebVitals;