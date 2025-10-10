### plot pred results

setwd(this.path::here())
load("sim_results_from_deepspat.rda")

# ------------------------------------------------------------------------------
library(ggpubr)
library(ggnewscale)
library(RColorBrewer)
library(viridis)
library(grid)
library(gridExtra)
# pic_path = "Pic_GP_sim/"
# if (!dir.exists(pic_path)) {dir.create(pic_path)}

width1 = 11.5
unit.w1 = unit(width1, "cm")
width2 = 8
unit.w2 = unit(width2, "cm")
height1 = 8
unit.h1 = unit(height1, "cm")
ref_shap = 21
ref_shap1 = 8
axis.title.size = 16
axis.text.size = 16
legend.text.size = 15
legend.title.size = 16
text.size = 5
# ------------------------------------------------------------------------------

# 1
plot_mean_gstat <- ggplot(pred_gstat@data) +
  geom_point(aes(pred_gstat@coords[,1], pred_gstat@coords[,2], color = var1.pred), 
             size = 0.5) +
  scale_color_distiller(palette = "Spectral", 
                        name = expression(E(bold(Z)[test] ~ "|" ~ bold(Z))),
                        limits = c(-4,4), 
                        oob = scales::squish, 
                        guide = "none") +
  theme_bw() + coord_fixed() + theme(text = element_text(size=15)) +
  labs(x = expression(s[1]), y = expression(s[2])) +
  theme(plot.title = element_text(hjust = 0.5, size=25),
        axis.title=element_text(size=axis.title.size),
        axis.text = element_text(size=axis.text.size),
        axis.line = element_blank(),
        axis.ticks = element_blank(),
        legend.key.size = unit(0.25, "in"),
        legend.text = element_text(size=legend.text.size),
        legend.title = element_text(size=legend.title.size, margin = margin(b = 10)),
        # legend.position = "right", #c(0.18, 0.8),
        legend.background = element_rect(fill='transparent'), #alpha('white', 0.4)
        legend.direction = "vertical",
        legend.box = "vertical",
        legend.spacing.y = unit(0.4, "lines"),
        legend.margin = margin(5, 5, 5, 5))

# 2
plot_mean_gp <- ggplot(pred_gp$df_pred) +
  geom_point(aes(x, y, color = pred_mean), size = 0.5) +
  scale_color_distiller(palette = "Spectral", name = expression(E(bold(Z)[test] ~ "|" ~ bold(Z))),
                        limits = c(-4,4), oob = scales::squish, guide = "none") +
  theme_bw() + coord_fixed() + theme(text = element_text(size=15)) +
  labs(x = expression(s[1]), y = expression(s[2])) +
  theme(plot.title = element_text(hjust = 0.5, size=25),
        axis.title=element_text(size=axis.title.size),
        axis.text = element_text(size=axis.text.size),
        axis.line = element_blank(),
        axis.ticks = element_blank(),
        legend.key.size = unit(0.25, "in"),
        legend.text = element_text(size=legend.text.size),
        legend.title = element_text(size=legend.title.size, margin = margin(b = 10)),
        # legend.position = "right", #c(0.18, 0.8),
        legend.background = element_rect(fill='transparent'), #alpha('white', 0.4)
        legend.direction = "vertical",
        legend.box = "vertical",
        legend.spacing.y = unit(0.4, "lines"),
        legend.margin = margin(5, 5, 5, 5))

# 3
plot_mean_cocons <- ggplot(data.frame(pred_cocons)) +
  geom_point(aes(pred_gp$df_pred$x, pred_gp$df_pred$y, color = mean + trend), size = 0.5) +
  scale_color_distiller(palette = "Spectral", name = expression(E(bold(Z)[test] ~ "|" ~ bold(Z))),
                        limits = c(-4,4), oob = scales::squish) +
  theme_bw() + coord_fixed() + theme(text = element_text(size=15))  +
  labs(x = expression(s[1]), y = expression(s[2])) +
  theme(plot.title = element_text(hjust = 0.5, size=25),
        axis.title=element_text(size=axis.title.size),
        axis.text = element_text(size=axis.text.size),
        axis.line = element_blank(),
        axis.ticks = element_blank(),
        legend.key.size = unit(0.25, "in"),
        legend.text = element_text(size=legend.text.size),
        legend.title = element_text(size=legend.title.size, margin = margin(b = 10)),
        # legend.position = "right", #c(0.18, 0.8),
        legend.background = element_rect(fill='transparent'), #alpha('white', 0.4)
        legend.direction = "vertical",
        legend.box = "vertical",
        legend.spacing.y = unit(0.4, "lines"),
        legend.margin = margin(5, 5, 5, 5))

# 4
plot_sd_gstat <- ggplot(pred_gstat@data) +
  geom_point(aes(pred_gstat@coords[,1], pred_gstat@coords[,2], color = sqrt(var1.var) ), size = 0.5) +
  scale_color_distiller(palette = "BrBG", name = expression(sd(bold(Z)[test] ~ "|" ~ bold(Z))),
                        limits = c(0,0.6), oob = scales::squish, guide = "none") +
  theme_bw() + coord_fixed() + theme(text = element_text(size=15))  +
  labs(x = expression(s[1]), y = expression(s[2])) +
  theme(plot.title = element_text(hjust = 0.5, size=25),
        axis.title=element_text(size=axis.title.size),
        axis.text = element_text(size=axis.text.size),
        axis.line = element_blank(),
        axis.ticks = element_blank(),
        legend.key.size = unit(0.25, "in"),
        legend.text = element_text(size=legend.text.size),
        legend.title = element_text(size=legend.title.size, margin = margin(b = 10)),
        # legend.position = "right", #c(0.18, 0.8),
        legend.background = element_rect(fill='transparent'), #alpha('white', 0.4)
        legend.direction = "vertical",
        legend.box = "vertical",
        legend.spacing.y = unit(0.4, "lines"),
        legend.margin = margin(5, 5, 5, 5))

# 5
plot_sd_gp <-ggplot(pred_gp$df_pred) +
  geom_point(aes(x, y, color = sqrt(pred_var)), size = 0.5) + # + as.numeric(1/d_gp$precy_tf) ) ), size = 0.5) +
  scale_color_distiller(palette = "BrBG", name = expression(sd(bold(Z)[test] ~ "|" ~ bold(Z))),
                        limits = c(0,0.6), oob = scales::squish, guide = "none") +
  theme_bw() + coord_fixed() + theme(text = element_text(size=15)) +
  labs(x = expression(s[1]), y = expression(s[2])) +
  theme(plot.title = element_text(hjust = 0.5, size=25),
        axis.title=element_text(size=axis.title.size),
        axis.text = element_text(size=axis.text.size),
        axis.line = element_blank(),
        axis.ticks = element_blank(),
        legend.key.size = unit(0.25, "in"),
        legend.text = element_text(size=legend.text.size),
        legend.title = element_text(size=legend.title.size, margin = margin(b = 10)),
        # legend.position = "right", #c(0.18, 0.8),
        legend.background = element_rect(fill='transparent'), #alpha('white', 0.4)
        legend.direction = "vertical",
        legend.box = "vertical",
        legend.spacing.y = unit(0.4, "lines"),
        legend.margin = margin(5, 5, 5, 5))

# 6
plot_sd_cocons <-ggplot(data.frame(pred_cocons)) +
  geom_point(aes(pred_gp$df_pred$x, pred_gp$df_pred$y, color = sd.pred ), size = 0.5) +
  scale_color_distiller(palette = "BrBG", name = expression(sd(bold(Z)[test] ~ "|" ~ bold(Z))),
                        limits = c(0,0.6), oob = scales::squish) +
  theme_bw() + coord_fixed() + theme(text = element_text(size=15))  +
  labs(x = expression(s[1]), y = expression(s[2])) +
  theme(plot.title = element_text(hjust = 0.5, size=25),
        axis.title=element_text(size=axis.title.size),
        axis.text = element_text(size=axis.text.size),
        axis.line = element_blank(),
        axis.ticks = element_blank(),
        legend.key.size = unit(0.25, "in"),
        legend.text = element_text(size=legend.text.size),
        legend.title = element_text(size=legend.title.size, margin = margin(b = 10)),
        # legend.position = "right", #c(0.18, 0.8),
        legend.background = element_rect(fill='transparent'), #alpha('white', 0.4)
        legend.direction = "vertical",
        legend.box = "vertical",
        legend.spacing.y = unit(0.4, "lines"),
        legend.margin = margin(5, 5, 5, 5))

library(patchwork)

plots_list <- list(plot_mean_gstat, plot_mean_gp, plot_mean_cocons,
                   plot_sd_gstat, plot_sd_gp, plot_sd_cocons)
plot_sim_deepspat <- wrap_plots(plots_list, nrow = 2)
plot_sim_deepspat

ggsave(filename="plot_sims_from_deepspat.png", plot=plot_sim_deepspat,
       device="png", width=30, height=20, scale=1, units="cm", dpi=300)

