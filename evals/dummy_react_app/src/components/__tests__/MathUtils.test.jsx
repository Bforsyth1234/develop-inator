import { describe, it, expect } from "vitest";
import { safeDivide, factorial, clamp } from "../MathUtils";

describe("MathUtils", () => {
  describe("safeDivide", () => {
    it("divides two numbers", () => {
      expect(safeDivide(10, 2)).toBe(5);
    });

    it("handles decimal results", () => {
      expect(safeDivide(7, 2)).toBe(3.5);
    });

    // NOTE: This test documents the existing bug — it will need
    // updating when the divide-by-zero fix is applied.
    it("returns Infinity when dividing by zero (known bug)", () => {
      expect(safeDivide(10, 0)).toBe(Infinity);
    });
  });

  describe("factorial", () => {
    it("returns 1 for 0", () => {
      expect(factorial(0)).toBe(1);
    });

    it("computes 5!", () => {
      expect(factorial(5)).toBe(120);
    });

    it("returns -1 for negative input", () => {
      expect(factorial(-3)).toBe(-1);
    });
  });

  describe("clamp", () => {
    it("clamps above max", () => {
      expect(clamp(15, 0, 10)).toBe(10);
    });

    it("clamps below min", () => {
      expect(clamp(-5, 0, 10)).toBe(0);
    });

    it("returns value when in range", () => {
      expect(clamp(5, 0, 10)).toBe(5);
    });
  });
});

