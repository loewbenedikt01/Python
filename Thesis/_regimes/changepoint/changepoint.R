

library(arrow)
library(cpm)
library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)

# ---------------------------------------------------------------------------
# 1. Load and clean
# ---------------------------------------------------------------------------
returns <- read_parquet("C:/Users/benel/Coding/Python/Thesis/_database/changepoint.parquet")

# FIX 1: format() first, so a tz-aware timestamp can't shift the date by a day
returns <- returns %>%
  mutate(date = as.Date(format(Date, "%Y-%m-%d"))) %>%
  select(-Date)

returns_filtered <- returns %>%
  filter(date >= as.Date("1990-01-03"), date <= as.Date("2025-12-31"))

# ---------------------------------------------------------------------------
# 2. Detection
# ---------------------------------------------------------------------------
detect_changepoints <- function(df, target_ticker, cpmType = "Mood",
                                ARL0 = 10000, startup = 20) {
  
  # FIX 2: exact match. grep("VIX", fixed=TRUE) would also hit ^VIX3M, VIXY, etc.
  matched_col <- which(names(df) == target_ticker)
  
  if (length(matched_col) != 1L) {
    stop(paste("Expected exactly one column named", target_ticker,
               "- found", length(matched_col), "in:", paste(names(df), collapse = ", ")))
  }
  
  clean_df <- df %>%
    select(date, value = all_of(matched_col)) %>%
    filter(!is.na(value))
  
  res <- processStream(clean_df$value, cpmType = cpmType, ARL0 = ARL0, startup = startup)
  
  data.frame(
    detection_time    = res$detectionTimes,
    changepoint_index = res$changePoints,
    changepoint_date  = clean_df$date[res$changePoints],
    detection_date    = clean_df$date[res$detectionTimes]
  )
}

vix_cps  <- detect_changepoints(returns_filtered, "^VIX",  cpmType = "Mood", ARL0 = 10000)
gspc_cps <- detect_changepoints(returns_filtered, "^GSPC", cpmType = "Mood", ARL0 = 10000)

# sanity: the paper reports 27 change points in each series through Sep 2015
cat("VIX  breaks:", nrow(vix_cps),
    "| through 2015-09-30:", sum(vix_cps$detection_date  <= as.Date("2015-09-30")), "\n")
cat("GSPC breaks:", nrow(gspc_cps),
    "| through 2015-09-30:", sum(gspc_cps$detection_date <= as.Date("2015-09-30")), "\n")

write_csv(vix_cps,  "C:/Users/benel/Coding/Python/Thesis/_regimes/changepoint/vix_changepoint.csv")
write_csv(gspc_cps, "C:/Users/benel/Coding/Python/Thesis/_regimes/changepoint/gspc_changepoint.csv")

# ---------------------------------------------------------------------------
# 3. Figure 4 replication
# ---------------------------------------------------------------------------
create_regime_shading <- function(cp_dates, start_date, end_date) {
  boundaries <- sort(unique(c(start_date, cp_dates, end_date)))
  regimes <- data.frame(
    xmin      = boundaries[-length(boundaries)],
    xmax      = boundaries[-1],
    regime_id = seq_len(length(boundaries) - 1)
  )
  regimes %>% filter(regime_id %% 2 == 0)
}

start_date <- min(returns_filtered$date)
end_date   <- max(returns_filtered$date)

vix_shading  <- create_regime_shading(vix_cps$changepoint_date,  start_date, end_date)
gspc_shading <- create_regime_shading(gspc_cps$changepoint_date, start_date, end_date)
vix_shading$Ticker  <- "^VIX"
gspc_shading$Ticker <- "^GSPC"
all_shading <- rbind(vix_shading, gspc_shading)

plot_data <- returns_filtered %>%
  select(date, `^VIX`, `^GSPC`) %>%
  pivot_longer(cols = c(`^VIX`, `^GSPC`), names_to = "Ticker", values_to = "Return")

ggplot() +
  geom_rect(data = all_shading,
            aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf),
            fill = "grey80", alpha = 0.5) +
  geom_line(data = plot_data, aes(x = date, y = Return, color = Ticker),
            linewidth = 0.4) +
  facet_wrap(~ Ticker, ncol = 1, scales = "free_y") +
  scale_color_manual(values = c("^GSPC" = "#1f77b4", "^VIX" = "#d62728")) +
  theme_minimal() +
  labs(
    title    = "Log Returns with Alternating Regime Shading",
    subtitle = "Grey bands highlight every second detected CPM regime (Mood Test)",
    x = "Date",
    y = "Log Return"
  ) +
  theme(
    legend.position = "none",
    strip.text = element_text(face = "bold", size = 11)
  )

