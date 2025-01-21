P = BPMmatlab.model;

% This example shows the propagation of an LP mode in a multimode fiber.
% The MMF is divided into four segments, where the first and last are
% straight, the second is bent in the x direction and the third is bent in
% the y direction. Plotting of the mode overlaps has been enabled by
% setting P.calcModeOverlaps = true. Due to symmetry, the modes with odd
% parity are not excited in the first two segments despite the bending.

% We use the getLabeledModeIdx() function to find out which index in the
% P.modes array correponds to a particular mode label (LP01, LP11e, etc.)

%% General and solver-related settings
P.name = mfilename;
P.useAllCPUs = true; % If false, BPM-Matlab will leave one processor unused. Useful for doing other work on the PC while simulations are running.
P.useGPU = false; % (Default: false) Use CUDA acceleration for NVIDIA GPUs

%% Visualization parameters
P.calcModeOverlaps = true;  % Set it to true to calculate mode overlap integrals of propagating field with respect to different modes in the P.modes struct array
updatestepsize = 1e-5;

%% Resolution-related parameters (check for convergence)
P.Lx_main = 30e-6;        % [m] x side length of main area
P.Ly_main = 30e-6;        % [m] y side length of main area
P.Nx_main = 150;          % x resolution of main area
P.Ny_main = 150;          % y resolution of main area
P.padfactor = 1.5;  % How much absorbing padding to add on the sides of the main area (1 means no padding, 2 means the absorbing padding on both sides is of thickness Lx_main/2)
P.dz_target = 0.5e-6; % [m] z step size to aim for
P.alpha = 3e14;             % [1/m^3] "Absorption coefficient" per squared unit length distance out from edge of main area

%% Problem definition
P.lambda = 800e-9; % [m] Wavelength
P.n_background = 1.45; % [] (may be complex) Background refractive index, (in this case, the cladding)
P.n_0 = 1.4666; % [] reference refractive index

P = initializeRIfromFunction(P,@calcRI);

%% Segment 1
P.Lz = 5e-4; % [m] z propagation distances for this segment
P.updates = P.Lz/updatestepsize;
P.bendDirection = 0; % [degrees] direction of the bending, in a polar coordinate system with 0° to the right (towards positive x) and increasing angles in counterclockwise direction
P.bendingRoC = Inf; % [m] radius of curvature of the bend

P = findModes(P,30); % Find up to 30 modes

% modeIdx = getLabeledModeIndex(P,'LP01'); % The LP01 mode, relatively sensitive to bending-induced mode coupling
modeIdx = getLabeledModeIndex(P,'LP11e'); % The LP11e mode, about equally sensitive as LP01
% modeIdx = getLabeledModeIndex(P,'LP04'); % The LP04 mode, less sensitive

P.E = P.modes(modeIdx);

P = FD_BPM(P);

%% Segment 2
P.Lz = 1e-3; % [m] z propagation distances for this segment
P.updates = P.Lz/updatestepsize;
P.bendDirection = 0;
P.bendingRoC = 10e-3;

P = FD_BPM(P);

%% Segment 3
P.Lz = 1e-3; % [m] z propagation distances for this segment
P.updates = P.Lz/updatestepsize;
P.bendDirection = 90;
P.bendingRoC = 10e-3;

P = FD_BPM(P);

%% Segment 4
P.Lz = 5e-4; % [m] z propagation distances for this segment
P.updates = P.Lz/updatestepsize;
P.bendDirection = 0;
P.bendingRoC = Inf;

P = FD_BPM(P);

%% USER DEFINED RI FUNCTIONS
function n = calcRI(X,Y,n_background,nParameters)
% n may be complex
n = n_background*ones(size(X)); % Start by setting all pixels to n_background
n(X.^2 + Y.^2 < 6e-6^2) = 1.4666;
end