/**
 * useCalibration Hook
 *
 * Manages calibration workflow state and logic
 * Reference: Python implementation at python/demo/main.py:195-277
 *
 * Workflow:
 * 1. Display instructions
 * 2. For each calibration point:
 *    - Show animated dot
 *    - Collect 20-30 gaze samples
 *    - Apply statistical filtering
 * 3. Call tracker.adapt() with Python defaults
 * 4. Display completion message
 */

import React, { useState, useCallback, useRef } from 'react';
import { GazeResult } from 'webeyetrack';
import {
  CalibrationState,
  CalibrationSample,
  CalibrationPointData,
  CalibrationConfig,
  DEFAULT_CALIBRATION_POSITIONS,
  DEFAULT_CALIBRATION_CONFIG,
} from '../types/calibration';
import { filterSamples, prepareAdaptationData } from '../utils/calibrationHelpers';

interface UseCalibrationProps {
  /** WebEyeTrackProxy instance with adapt() method */
  tracker: any;  // WebEyeTrackProxy type

  /** Calibration configuration */
  config?: Partial<CalibrationConfig>;

  /** Callback when calibration completes successfully */
  onComplete?: () => void;

  /** Callback when calibration fails */
  onError?: (error: string) => void;
}

export function useCalibration({
  tracker,
  config: userConfig = {},
  onComplete,
  onError
}: UseCalibrationProps) {
  // Memoize config to prevent dependency changes
  const config = React.useMemo(
    () => ({ ...DEFAULT_CALIBRATION_CONFIG, ...userConfig }),
    [userConfig]
  );

  // Calibration state
  const [state, setState] = useState<CalibrationState>({
    status: 'idle',
    currentPointIndex: 0,
    totalPoints: config.numPoints,
    pointsData: []
  });

  // Reference to current sample collection
  const samplesRef = useRef<CalibrationSample[]>([]);
  const collectionTimerRef = useRef<NodeJS.Timeout | null>(null);

  /**
   * Handle incoming gaze results during sample collection
   */
  const handleGazeResult = useCallback((gazeResult: GazeResult) => {
    if (state.status !== 'collecting') return;

    const currentPoint = DEFAULT_CALIBRATION_POSITIONS[state.currentPointIndex];

    // Add sample to collection
    samplesRef.current.push({
      gazeResult,
      groundTruth: currentPoint,
      timestamp: Date.now()
    });

    console.log(`Collected sample ${samplesRef.current.length}/${config.samplesPerPoint} for point ${state.currentPointIndex + 1}`);
  }, [state.status, state.currentPointIndex, config.samplesPerPoint]);

  /**
   * Start calibration workflow
   */
  const startCalibration = useCallback(() => {
    if (!tracker) {
      const error = 'Tracker not initialized';
      console.error(error);
      if (onError) onError(error);
      return;
    }

    console.log('Starting calibration with config:', config);

    setState({
      status: 'instructions',
      currentPointIndex: 0,
      totalPoints: config.numPoints,
      pointsData: []
    });

    // Move to first calibration point after brief delay
    setTimeout(() => {
      setState(prev => ({
        ...prev,
        status: 'collecting'
      }));
    }, 3000);  // 3 second instruction display
  }, [tracker, config, onError]);

  /**
   * Handle animation completion (dot turned white)
   * Start collecting samples
   */
  const handleAnimationComplete = useCallback(() => {
    if (state.status !== 'collecting') return;

    console.log(`Animation complete for point ${state.currentPointIndex + 1}, starting sample collection`);

    // Clear previous samples
    samplesRef.current = [];

    // Stop collection after specified duration
    collectionTimerRef.current = setTimeout(() => {
      finishCurrentPoint();
    }, config.collectionDuration);
  }, [state.status, state.currentPointIndex, config.collectionDuration]);

  /**
   * Finish collecting samples for current point
   * Apply filtering and move to next point or complete calibration
   */
  const finishCurrentPoint = useCallback(async () => {
    if (collectionTimerRef.current) {
      clearTimeout(collectionTimerRef.current);
      collectionTimerRef.current = null;
    }

    const currentSamples = [...samplesRef.current];
    const currentPoint = DEFAULT_CALIBRATION_POSITIONS[state.currentPointIndex];

    console.log(`Finishing point ${state.currentPointIndex + 1}, collected ${currentSamples.length} samples`);

    // Apply statistical filtering (matching Python main.py:217-238)
    const filteredSample = filterSamples(currentSamples);

    if (!filteredSample) {
      const error = `Failed to collect samples for point ${state.currentPointIndex + 1}`;
      console.error(error);
      setState(prev => ({ ...prev, status: 'error', error }));
      if (onError) onError(error);
      return;
    }

    // Store point data
    const pointData: CalibrationPointData = {
      position: currentPoint,
      samples: currentSamples,
      filteredSample
    };

    const newPointsData = [...state.pointsData, pointData];

    // Check if this was the last point
    if (state.currentPointIndex + 1 >= config.numPoints) {
      // All points collected, perform adaptation
      await performAdaptation(newPointsData);
    } else {
      // Move to next point
      setState(prev => ({
        ...prev,
        currentPointIndex: prev.currentPointIndex + 1,
        pointsData: newPointsData,
        status: 'collecting'
      }));
    }
  }, [state, config.numPoints, onError]);

  /**
   * Perform model adaptation using collected calibration data
   * Reference: Python main.py:246-253
   */
  const performAdaptation = useCallback(async (pointsData: CalibrationPointData[]) => {
    console.log('Performing model adaptation with', pointsData.length, 'calibration points');

    setState(prev => ({ ...prev, status: 'processing' }));

    try {
      // Extract filtered samples
      const filteredSamples = pointsData
        .map(pd => pd.filteredSample)
        .filter(s => s !== undefined) as CalibrationSample[];

      if (filteredSamples.length < 3) {
        throw new Error(`Insufficient calibration points: ${filteredSamples.length} (minimum 3 required)`);
      }

      // Prepare data for tracker.adapt()
      const { eyePatches, headVectors, faceOrigins3D, normPogs } = prepareAdaptationData(filteredSamples);

      console.log('Calling tracker.adapt() with:');
      console.log('  - Points:', normPogs);
      console.log('  - stepsInner:', config.stepsInner, '(Python default, NOT 1)');
      console.log('  - innerLR:', config.innerLR);

      // Call adaptation with Python default parameters
      // CRITICAL: Use stepsInner=5 (Python default), NOT 1 (JS default)
      await tracker.adapt(
        eyePatches,
        headVectors,
        faceOrigins3D,
        normPogs,
        config.stepsInner,  // 5 (Python default)
        config.innerLR,     // 1e-5
        'calib'             // Point type
      );

      console.log('Calibration adaptation complete!');

      setState(prev => ({
        ...prev,
        status: 'complete',
        pointsData
      }));

      if (onComplete) {
        onComplete();
      }

      // Auto-close after 2 seconds
      setTimeout(() => {
        resetCalibration();
      }, 2000);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Calibration failed';
      console.error('Calibration error:', errorMessage);

      setState(prev => ({
        ...prev,
        status: 'error',
        error: errorMessage
      }));

      if (onError) {
        onError(errorMessage);
      }
    }
  }, [tracker, config, onComplete, onError]);

  /**
   * Cancel calibration and reset state
   */
  const cancelCalibration = useCallback(() => {
    if (collectionTimerRef.current) {
      clearTimeout(collectionTimerRef.current);
      collectionTimerRef.current = null;
    }

    resetCalibration();
  }, []);

  /**
   * Reset calibration state to idle
   */
  const resetCalibration = useCallback(() => {
    samplesRef.current = [];
    setState({
      status: 'idle',
      currentPointIndex: 0,
      totalPoints: config.numPoints,
      pointsData: []
    });
  }, [config.numPoints]);

  return {
    state,
    startCalibration,
    cancelCalibration,
    handleGazeResult,
    handleAnimationComplete,
    isCalibrating: state.status !== 'idle'
  };
}
