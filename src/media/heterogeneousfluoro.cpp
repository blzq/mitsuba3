#include <mitsuba/core/frame.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/volume.h>

NAMESPACE_BEGIN(mitsuba)


/**!

.. _medium-heterogeneousfluoro:

Heterogeneous fluorescent medium (:monosp:`heterogeneousfluoro`)
-----------------------------------------------

.. pluginparameters::

 * - albedo
   - |float|, |spectrum| or |volume|
   - Single-scattering albedo of the medium (Default: 0.75).
   - |exposed|, |differentiable|

 * - sigma_t
   - |float|, |spectrum| or |volume|
   - Extinction coefficient in inverse scene units (Default: 1).
   - |exposed|, |differentiable|

 * - scale
   - |float|
   - Optional scale factor that will be applied to the extinction parameter.
     It is provided for convenience when accommodating data based on different
     units, or to simply tweak the density of the medium. (Default: 1)
   - |exposed|

 * - sample_emitters
   - |bool|
   - Flag to specify whether shadow rays should be cast from inside the volume (Default: |true|)
     If the medium is enclosed in a :ref:`dielectric <bsdf-dielectric>` boundary,
     shadow rays are ineffective and turning them off will significantly reduce
     render time. This can reduce render time up to 50% when rendering objects
     with subsurface scattering.

 * - (Nested plugin)
   - |phase|
   - A nested phase function that describes the directional scattering properties of
     the medium. When none is specified, the renderer will automatically use an instance of
     isotropic.
   - |exposed|, |differentiable|


TODO

.. tabs::
    .. code-tab:: xml
        :name: lst-heterogeneousfluoro

        <!-- Declare a heterogeneous fluorescent participating medium named 'smoke' -->
        <medium type="heterogeneousfluoro" id="smoke">
            <!-- Acquire extinction values from an external data file -->
            <volume name="sigma_t" type="gridvolume">
                <string name="filename" value="frame_0150.vol"/>
            </volume>

            <!-- The albedo is constant and set to 0.9 -->
            <float name="albedo" value="0.9"/>

            <!-- Use an isotropic phase function -->
            <phase type="isotropic"/>

            <!-- Scale the density values as desired -->
            <float name="scale" value="200"/>
        </medium>

        <!-- Attach the index-matched medium to a shape in the scene -->
        <shape type="obj">
            <!-- Load an OBJ file, which contains a mesh version
                 of the axis-aligned box of the volume data file -->
            <string name="filename" value="bounds.obj"/>

            <!-- Reference the medium by ID -->
            <ref name="interior" id="smoke"/>
            <!-- If desired, this shape could also declare
                a BSDF to create an index-mismatched
                transition, e.g.
                <bsdf type="dielectric"/>
            -->
        </shape>

    .. code-tab:: python

        # Declare a heterogeneous fluorescent participating medium named 'smoke'
        'smoke': {
            'type': 'heterogeneousfluoro',

            # Acquire extinction values from an external data file
            'sigma_t': {
                'type': 'gridvolume',
                'filename': 'frame_0150.vol'
            },

            # The albedo is constant and set to 0.9
            'albedo': 0.9,

            # Use an isotropic phase function
            'phase': {
                'type': 'isotropic'
            },

            # Scale the density values as desired
            'scale': 200
        },

        # Attach the index-matched medium to a shape in the scene
        'shape': {
            'type': 'obj',
            # Load an OBJ file, which contains a mesh version
            # of the axis-aligned box of the volume data file
            'filename': 'bounds.obj',

            # Reference the medium by ID
            'interior': 'smoke',
            # If desired, this shape could also declare
            # a BSDF to create an index-mismatched
            # transition, e.g.
            # 'bsdf': {
            #     'type': 'isotropic'
            # },
        }
*/
template <typename Float, typename Spectrum>
class HeterogeneousFluoroMedium final : public Medium<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Medium, m_is_homogeneous, m_has_spectral_extinction,
                   m_has_fluorescence, m_phase_function)
    MI_IMPORT_TYPES(Scene, Sampler, Texture, Volume)

    HeterogeneousFluoroMedium(const Properties &props) : Base(props) {
        m_is_homogeneous = false;
        m_has_fluorescence = true;

        m_albedo = props.get_volume<Volume>("albedo", .3f);
        m_excitation = props.get_volume<Volume>("excitation", .3f);
        m_sigmaf = props.get_volume<Volume>("fluorescence", .5f);
        // Absorption + scattering + fluorescent excitation
        m_sigmat = props.get_volume<Volume>("m_sigmat", 1.0f);
        
        m_scale = props.get<ScalarFloat>("scale", 1.0f);
        m_has_spectral_extinction = props.get<bool>("has_spectral_extinction", true);

        m_max_density = dr::opaque<Float>(m_scale * (m_sigmat->max() + m_sigmaf->max()));
    }

    void traverse(TraversalCallback *cb) override {
        cb->put("scale",        m_scale,        ParamFlags::NonDifferentiable);
        cb->put("albedo",       m_albedo,       ParamFlags::Differentiable);
        cb->put("sigma_t",      m_sigmat,       ParamFlags::Differentiable);
        cb->put("excitation",   m_excitation,   ParamFlags::Differentiable);
        cb->put("fluorescence", m_sigmaf,       ParamFlags::Differentiable);
        Base::traverse(cb);
    }

    void parameters_changed(const std::vector<std::string> & /* keys */ = {}) override {
        m_max_density = dr::opaque<Float>(m_scale * (m_sigmat->max() + m_sigmaf->max()));
    }

    UnpolarizedSpectrum
    get_majorant(const MediumInteraction3f & /* mi */,
                 Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::MediumEvaluate, active);
        return m_max_density;
    }

    std::tuple<UnpolarizedSpectrum, UnpolarizedSpectrum, UnpolarizedSpectrum>
    get_scattering_coefficients(const MediumInteraction3f &mi,
                                Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::MediumEvaluate, active);

        auto sigmat = m_scale * (m_sigmat->eval(mi, active));
        auto sigmaf = m_scale * (m_sigmaf->eval(mi, active));
        if (has_flag(m_phase_function->flags(), PhaseFunctionFlags::Microflake)) {
            sigmat *= m_phase_function->projected_area(mi, active);
            sigmaf *= m_phase_function->projected_area(mi, active);
        }

        // Only the portion of sigman that represents null scattering
        auto sigman_null = m_max_density - sigmat - sigmaf;
        auto sigmas = sigmat * m_albedo->eval(mi, active);
    
        return { sigmas, sigman_null, sigmat };
    }

    std::tuple<UnpolarizedSpectrum, UnpolarizedSpectrum, UnpolarizedSpectrum, UnpolarizedSpectrum>
    get_scattering_coefficients_fluoro(const MediumInteraction3f &mi,
                                       Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::MediumEvaluate, active);

        auto sigmat = m_scale * (m_sigmat->eval(mi, active));
        auto sigmaf = m_scale * (m_sigmaf->eval(mi, active));
        if (has_flag(m_phase_function->flags(), PhaseFunctionFlags::Microflake)) {
            sigmat *= m_phase_function->projected_area(mi, active);
            sigmaf *= m_phase_function->projected_area(mi, active);
        }

        // Only the portion of sigman that represents null scattering
        auto sigman_null = m_max_density - sigmat - sigmaf;
        auto sigmas = sigmat * m_albedo->eval(mi, active);
    
        return { sigmas, sigman_null, sigmat, sigmaf };
    }

    MediumInteraction3f sample_interaction(const Ray3f &ray, Float sample,
                                           UInt32 channel, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::MediumSample, active);

        // initialize basic medium interaction fields
        MediumInteraction3f mei = dr::zeros<MediumInteraction3f>();
        mei.wi          = -ray.d;
        mei.sh_frame    = Frame3f(mei.wi);
        mei.time        = ray.time;
        mei.wavelengths = ray.wavelengths;

        auto [aabb_its, mint, maxt] = intersect_aabb(ray);
        aabb_its &= (dr::isfinite(mint) || dr::isfinite(maxt));
        active &= aabb_its;
        dr::masked(mint, !active) = 0.f;
        dr::masked(maxt, !active) = dr::Infinity<Float>;

        mint = dr::maximum(0.f, mint);
        maxt = dr::minimum(ray.maxt, maxt);

        auto combined_extinction = get_majorant(mei, active);
        Float m                  = combined_extinction[0];
        if constexpr (is_rgb_v<Spectrum>) { // Handle RGB rendering
            dr::masked(m, channel == 1u) = combined_extinction[1];
            dr::masked(m, channel == 2u) = combined_extinction[2];
        } else {
            DRJIT_MARK_USED(channel);
        }

        Float sampled_t = mint + (-dr::log(1 - sample) / m);
        Mask valid_mi   = active && (sampled_t <= maxt);
        mei.t           = dr::select(valid_mi, sampled_t, dr::Infinity<Float>);
        mei.p           = ray(sampled_t);
        mei.medium      = this;
        mei.mint        = mint;

        std::tie(
            mei.sigma_s, mei.sigma_n, mei.sigma_t, mei.sigma_f
        ) = get_scattering_coefficients_fluoro(mei, valid_mi);
        mei.combined_extinction = combined_extinction;
        return mei;
    }

    std::pair<Wavelength, UnpolarizedSpectrum>
    sample_wavelength_shift(const MediumInteraction3f &mi,
                            Float sample, Mask active) const override {
        // Only support paths from camera to light (TransportMode::Radiance)
        auto [wavelengths, weight] = m_excitation->sample_spectrum(
            mi, math::sample_shifted<Wavelength>(sample), active);

        return { wavelengths, mi.sigma_t * weight };
    }

    std::tuple<Mask, Float, Float>
    intersect_aabb(const Ray3f &ray) const override {
        return m_sigmat->bbox().ray_intersect(ray);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "HeterogeneousFluoroMedium[" << std::endl
            << "  albedo       = " << string::indent(m_albedo) << std::endl
            << "  excitation   = " << string::indent(m_excitation) << std::endl
            << "  fluorescence = " << string::indent(m_sigmaf) << std::endl
            << "  sigma_t      = " << string::indent(m_sigmat) << std::endl
            << "  scale        = " << string::indent(m_scale) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(HeterogeneousFluoroMedium)
private:
    ref<Volume> m_albedo, m_excitation, m_sigmat, m_sigmaf;
    ScalarFloat m_scale;
    Float m_max_density;

    MI_TRAVERSE_CB(Base, m_sigmat, m_albedo, m_excitation, m_sigmaf, m_max_density)
};

MI_EXPORT_PLUGIN(HeterogeneousFluoroMedium)
NAMESPACE_END(mitsuba)
