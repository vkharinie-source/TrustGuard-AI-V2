// Centralized API client for TrustGuard AI

const BASE_URL =
  import.meta.env.VITE_API_URL ||
  "http://127.0.0.1:8000";

const getHeaders = (isJson = true) => {
  const headers = {};

  if (isJson) {
    headers["Content-Type"] = "application/json";
  }

  const token = localStorage.getItem("trustguard_token");

  if (token) {
    headers["Authorization"] = "Bearer " + token;
  }

  return headers;
};

export const api = {

  // =========================================================
  // AUTHENTICATION
  // =========================================================

  login: async (email, password) => {
    const res = await fetch(
      BASE_URL + "/api/auth/login",
      {
        method: "POST",
        headers: getHeaders(true),
        body: JSON.stringify({
          email: email,
          password: password
        })
      }
    );

    const data = await res.json();

    if (!res.ok) {
      throw new Error(
        data.detail ||
        data.message ||
        "Login failed."
      );
    }

    if (data.token) {
      localStorage.setItem(
        "trustguard_token",
        data.token
      );

      localStorage.setItem(
        "trustguard_user",
        JSON.stringify(
          data.user || {
            email: email
          }
        )
      );
    }

    return data;
  },

  register: async (
    fullName,
    email,
    password,
    confirmPassword
  ) => {
    const res = await fetch(
      BASE_URL + "/api/auth/register",
      {
        method: "POST",
        headers: getHeaders(true),
        body: JSON.stringify({
          full_name: fullName,
          fullName: fullName,
          email: email,
          password: password,
          confirm_password: confirmPassword,
          confirmPassword: confirmPassword
        })
      }
    );

    const data = await res.json();

    if (!res.ok) {
      throw new Error(
        data.detail ||
        data.message ||
        "Registration failed."
      );
    }

    return data;
  },

  // =========================================================
  // IMAGE PREDICTION
  // =========================================================

  predictImage: async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    let res = await fetch(
      BASE_URL + "/predict",
      {
        method: "POST",
        body: formData
      }
    );

    if (!res.ok && res.status === 404) {
      const fallbackFormData = new FormData();
      fallbackFormData.append("file", file);
      res = await fetch(
        BASE_URL + "/api/predict/image",
        {
          method: "POST",
          body: fallbackFormData
        }
      );
    }

    const data = await res.json();

    if (!res.ok) {
      throw new Error(
        data.detail ||
        data.message ||
        "Image prediction failed."
      );
    }

    return data;
  },

  // =========================================================
  // VIDEO PREDICTION (FRAME-LEVEL TIMELINE)
  // =========================================================

  predictVideo: async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(
      BASE_URL + "/api/predict/video",
      {
        method: "POST",
        body: formData
      }
    );

    const data = await res.json();

    if (!res.ok) {
      throw new Error(
        data.detail ||
        data.message ||
        "Video verification failed."
      );
    }

    return data;
  },

  // =========================================================
  // AUDIO PREDICTION (VOICE-CLONING / SYNTHETIC SPEECH)
  // =========================================================

  predictAudio: async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(
      BASE_URL + "/api/predict/audio",
      {
        method: "POST",
        body: formData
      }
    );

    const data = await res.json();

    if (!res.ok) {
      throw new Error(
        data.detail ||
        data.message ||
        "Audio authenticity verification failed."
      );
    }

    return data;
  },

  // =========================================================
  // ERROR LEVEL ANALYSIS (FORENSICS)
  // =========================================================

  runForensicsELA: async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(
      BASE_URL + "/api/forensics/ela",
      {
        method: "POST",
        body: formData
      }
    );

    const data = await res.json();

    if (!res.ok) {
      throw new Error(
        data.detail ||
        data.message ||
        "Forensic ELA analysis failed."
      );
    }

    return data;
  },

  // =========================================================
  // TRANSPARENCY & DATASET STATS
  // =========================================================

  getTransparencyStats: async () => {
    const res = await fetch(
      BASE_URL + "/api/transparency/stats",
      {
        headers: getHeaders(false)
      }
    );

    const data = await res.json();

    if (!res.ok) {
      throw new Error(
        data.detail ||
        data.message ||
        "Unable to fetch transparency statistics."
      );
    }

    return data;
  },


  analyzeWebsite: async (url) => {
    const res = await fetch(
      BASE_URL + "/api/analyze/website",
      {
        method: "POST",
        headers: getHeaders(true),
        body: JSON.stringify({
          url: url
        })
      }
    );

    const data = await res.json();

    if (!res.ok) {
      throw new Error(
        data.detail ||
        data.message ||
        "Website analysis failed."
      );
    }

    return data;
  },

  // =========================================================
  // CALL ANALYSIS
  // =========================================================

  analyzeCall: async (
    caller,
    transcript
  ) => {
    const res = await fetch(
      BASE_URL + "/api/analyze/call",
      {
        method: "POST",
        headers: getHeaders(true),
        body: JSON.stringify({
          caller: caller,
          transcript: transcript
        })
      }
    );

    const data = await res.json();

    if (!res.ok) {
      throw new Error(
        data.detail ||
        data.message ||
        "Call analysis failed."
      );
    }

    return data;
  },

  // =========================================================
  // EMAIL ANALYSIS
  // =========================================================

  analyzeEmail: async (
    subject,
    sender,
    content
  ) => {
    const res = await fetch(
      BASE_URL + "/api/analyze/email",
      {
        method: "POST",
        headers: getHeaders(true),
        body: JSON.stringify({
          subject: subject,
          sender: sender,
          content: content
        })
      }
    );

    const data = await res.json();

    if (!res.ok) {
      throw new Error(
        data.detail ||
        data.message ||
        "Email analysis failed."
      );
    }

    return data;
  },

  // =========================================================
  // MESSAGE ANALYSIS
  // =========================================================

  analyzeMessage: async (content) => {
    const res = await fetch(
      BASE_URL + "/api/analyze/message",
      {
        method: "POST",
        headers: getHeaders(true),
        body: JSON.stringify({
          content: content
        })
      }
    );

    const data = await res.json();

    if (!res.ok) {
      throw new Error(
        data.detail ||
        data.message ||
        "Message analysis failed."
      );
    }

    return data;
  },

  // =========================================================
  // APPLICATION ANALYSIS
  // =========================================================

  analyzeApplication: async (
    appName,
    developer,
    description
  ) => {
    try {
      const res = await fetch(
        BASE_URL + "/api/analyze/app",
        {
          method: "POST",
          headers: getHeaders(true),
          body: JSON.stringify({
            app_name: appName,
            developer: developer,
            description: description
          })
        }
      );

      if (res.ok) {
        return await res.json();
      }
    } catch (error) {
      // Use fallback
    }

    const res = await fetch(
      BASE_URL + "/api/analyze/message",
      {
        method: "POST",
        headers: getHeaders(true),
        body: JSON.stringify({
          content:
            "App: " +
            appName +
            ". Developer: " +
            developer +
            ". Details: " +
            description
        })
      }
    );

    const data = await res.json();

    if (!res.ok) {
      throw new Error(
        data.detail ||
        data.message ||
        "Application analysis failed."
      );
    }

    return data;
  },

  // =========================================================
  // INTERNSHIP ANALYSIS
  // =========================================================

  analyzeInternship: async (
    title,
    company,
    description
  ) => {
    try {
      const res = await fetch(
        BASE_URL + "/api/analyze/internship",
        {
          method: "POST",
          headers: getHeaders(true),
          body: JSON.stringify({
            title: title,
            company: company,
            description: description
          })
        }
      );

      if (res.ok) {
        return await res.json();
      }
    } catch (error) {
      // Use fallback
    }

    const res = await fetch(
      BASE_URL + "/api/analyze/email",
      {
        method: "POST",
        headers: getHeaders(true),
        body: JSON.stringify({
          subject: title,
          sender: company,
          content: description
        })
      }
    );

    const data = await res.json();

    if (!res.ok) {
      throw new Error(
        data.detail ||
        data.message ||
        "Internship analysis failed."
      );
    }

    return data;
  },

  // =========================================================
  // JOB ANALYSIS
  // =========================================================

  analyzeJob: async (
    title,
    company,
    description
  ) => {
    try {
      const res = await fetch(
        BASE_URL + "/api/analyze/job",
        {
          method: "POST",
          headers: getHeaders(true),
          body: JSON.stringify({
            title: title,
            company: company,
            description: description
          })
        }
      );

      if (res.ok) {
        return await res.json();
      }
    } catch (error) {
      // Use fallback
    }

    const res = await fetch(
      BASE_URL + "/api/analyze/message",
      {
        method: "POST",
        headers: getHeaders(true),
        body: JSON.stringify({
          content:
            "Job Title: " +
            title +
            ". Company: " +
            company +
            ". Description: " +
            description
        })
      }
    );

    const data = await res.json();

    if (!res.ok) {
      throw new Error(
        data.detail ||
        data.message ||
        "Job offer analysis failed."
      );
    }

    return data;
  },

  // =========================================================
  // NEWS VERIFICATION
  // =========================================================

  analyzeNews: async (
    headline,
    text
  ) => {
    try {
      const res = await fetch(
        BASE_URL + "/api/analyze/news",
        {
          method: "POST",
          headers: getHeaders(true),
          body: JSON.stringify({
            headline: headline,
            content: text,
            text: text
          })
        }
      );

      if (res.ok) {
        return await res.json();
      }
    } catch (error) {
      // Fallback to message route if news route is unavailable
    }

    const res = await fetch(
      BASE_URL + "/api/analyze/message",
      {
        method: "POST",
        headers: getHeaders(true),
        body: JSON.stringify({
          content:
            "News Headline: " +
            headline +
            ". Text: " +
            text
        })
      }
    );

    const data = await res.json();

    if (!res.ok) {
      throw new Error(
        data.detail ||
        data.message ||
        "News verification failed."
      );
    }

    return data;
  },

  // =========================================================
  // HISTORY
  // =========================================================

  getHistory: async () => {
    let lastError;

    const endpoints = [
      BASE_URL + "/api/history",
      BASE_URL + "/history"
    ];

    for (const endpoint of endpoints) {
      try {
        const res = await fetch(
          endpoint,
          {
            headers: getHeaders(false)
          }
        );

        if (!res.ok) {
          lastError = new Error(
            "History request failed (" +
            res.status +
            ")."
          );

          continue;
        }

        const data = await res.json();

        const items = Array.isArray(data)
          ? data
          : data.history;

        if (!Array.isArray(items)) {
          lastError = new Error(
            "The history response had an unexpected format."
          );

          continue;
        }

        return items;

      } catch (error) {
        lastError = error;
      }
    }

    throw (
      lastError ||
      new Error("Unable to load history.")
    );
  },

  // =========================================================
  // CLEAR HISTORY
  // =========================================================

  clearHistory: async () => {
    try {
      const res = await fetch(
        BASE_URL + "/api/history",
        {
          method: "DELETE",
          headers: getHeaders(false)
        }
      );

      return await res.json();

    } catch (error) {
      return {
        success: true
      };
    }
  },

  // =========================================================
  // DASHBOARD STATS
  // =========================================================

  getDashboardStats: async () => {
    try {
      const res = await fetch(
        BASE_URL + "/api/dashboard/stats",
        {
          headers: getHeaders(false)
        }
      );

      if (res.ok) {
        return await res.json();
      }

    } catch (error) {
      // Use history fallback
    }

    const items = await api.getHistory();

    let safeCount = 0;
    let threatCount = 0;
    let suspiciousCount = 0;

    items.forEach((item) => {
      const pred = String(
        item.prediction ||
        item.result ||
        ""
      ).toUpperCase();

      if (
        pred === "REAL" ||
        pred === "SAFE"
      ) {
        safeCount++;

      } else if (
        pred === "SUSPICIOUS"
      ) {
        suspiciousCount++;
        threatCount++;

      } else {
        threatCount++;
      }
    });

    return {
      success: true,

      stats: {
        total: items.length,
        safe: safeCount,
        suspicious: suspiciousCount,
        threats_detected: threatCount
      },

      recent: items.slice(0, 5)
    };
  },

  // =========================================================
  // HEALTH CHECK
  // =========================================================

  checkHealth: async () => {
    try {
      const res = await fetch(
        BASE_URL + "/health"
      );

      if (res.ok) {
        return await res.json();
      }

    } catch (error) {
      // Try next endpoint
    }

    try {
      const res = await fetch(
        BASE_URL + "/api/health"
      );

      if (res.ok) {
        return await res.json();
      }

    } catch (error) {
      // Backend unavailable
    }

    return {
      status: "offline"
    };
  }
};