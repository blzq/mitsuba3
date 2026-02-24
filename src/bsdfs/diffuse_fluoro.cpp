#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _bsdf-diffusefluoro:

Fluorescent diffuse material (:monosp:`diffusefluoro`)
-------------------------------------------

.. pluginparameters::

 * - reflectance
   - |spectrum| or |texture|
   - Specifies the diffuse albedo of the material (Default: 0.5)
   - |exposed|, |differentiable|
 * - fluorescence
   - |spectrum| or |texture|
   - Specifies the fluorescent emission albedo of the material (Default: 0.5)
   - |exposed|, |differentiable|
 * - excitation
   - |spectrum| or |texture|
   - Specifies the fluorescent excitation spectrum of the material (Default: 0.0)
   - |exposed|, |differentiable|

The fluorescent diffuse material represents an ideally diffuse (Lambertian)
material with reflectance and fluorescence components. Received illumination
is scattered so it looks the same independently of the direction of observation.
However, an incoming wavelength that is non-zero in the excitation spectrum has
a probability to be wavelength-shifted and re-emitted as one of the wavelengths
in the fluorescent emission spectrum.

Note that the input fluorescent emission spectrum will be normalised to have
a unit integral. This means that to increase the strength of fluorescent emission,
the excitation value should be increased instead. For energy conservation, the
sum of the excitation and reflectance spectra should not exceed 1.0 at any point
in the wavelength domain.

When using this plugin, you *must* enable one of the :monosp:`spectral` modes
of the renderer, as an error will be thrown otherwise. Also, to observe
fluorescent effects, a fluorescent-enabled integrator must be used.

Also note that this material is one-sided---that is, observed from the
back side, it will be completely black. If this is undesirable,
consider using the :ref:`twosided <bsdf-twosided>` BRDF adapter plugin.

.. tabs::
    .. code-tab:: xml
        :name: diffuse-fluoro-spectral

        <bsdf type="diffusefluoro">
            <spectrum name="reflectance" value="400:0.0, 500:0.3, 600:0.6, 700:0.0" />
		    <spectrum name="fluorescence" value="500:0.0, 600:0.5, 700:0.5" />
		    <spectrum name="excitation" value="400:0.4, 500:0.0" />
        </bsdf>

    .. code-tab:: python

        'type': 'diffusefluoro',
        'reflectance': {
            'type': 'irregular',
            'wavelengths': '400, 500, 600, 700',
            'values': '0.0, 0.3, 0.6, 0.0'
        },
        'fluorescence': {
            'type': 'irregular',
            'wavelengths': '500, 600, 700',
            'values': '0.0, 0.5, 0.5'
        },
        'excitation': {
            'type': 'irregular',
            'wavelengths': '400, 500',
            'values': '0.4, 0.0'
        }
*/
template <typename Float, typename Spectrum>
class SmoothDiffuseFluoro final : public BSDF<Float, Spectrum> {
public:
    MI_IMPORT_BASE(BSDF, m_flags, m_components)
    MI_IMPORT_TYPES(Texture)

    SmoothDiffuseFluoro(const Properties &props) : Base(props) {
        if constexpr (!is_spectral_v<Spectrum>) {
            Log(Error, "This BRDF can only be used in Mitsuba variants that "
                       "perform a spectral simulation.");
        }
        m_reflectance = props.get_texture<Texture>("reflectance", 0.f);
        m_fluorescence = props.get_texture<Texture>("fluorescence", .5f);
        m_excitation = props.get_texture<Texture>("excitation", .5f);

        m_components.push_back(BSDFFlags::DiffuseReflection | BSDFFlags::FrontSide);
        m_components.push_back(BSDFFlags::FluorescentReflection | BSDFFlags::FrontSide);

        m_flags = m_components[0] | m_components[1];
    }

    void traverse(TraversalCallback *cb) override {
        cb->put("reflectance",  m_reflectance,  ParamFlags::Differentiable);
        cb->put("fluorescence", m_fluorescence, ParamFlags::Differentiable);
        cb->put("excitation",   m_excitation,   ParamFlags::Differentiable);
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float sample1,
                                             const Point2f &sample2,
                                             Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        bool has_diffuse  = ctx.is_enabled(BSDFFlags::DiffuseReflection, 0),
             has_fluoro = ctx.is_enabled(BSDFFlags::FluorescentReflection, 1);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        BSDFSample3f bs = dr::zeros<BSDFSample3f>();

        active &= cos_theta_i > 0.f;

        if (unlikely((!has_diffuse && !has_fluoro) || dr::none_or<false>(active)))
            return { bs, 0.f };

        UnpolarizedSpectrum diffuse_value = m_reflectance->eval(si, active);
        UnpolarizedSpectrum fluoro_value = ctx.mode == TransportMode::Radiance
            ? m_fluorescence->eval_norm(si, active)
            : m_excitation->eval(si, active);

        Float prob_diffuse = 1.f;
        Float fluoro_scale = ctx.mode == TransportMode::Radiance
            ? m_excitation->sum(si, active) : 1.f;
        if (unlikely(has_fluoro != has_diffuse))
            prob_diffuse = has_diffuse ? 1.f : 0.f;
        else
            // TODO: bad approximation (?) when the number of wavelengths per ray > 1
            prob_diffuse = dr::mean(
                diffuse_value /
                (diffuse_value + fluoro_scale * fluoro_value + dr::Epsilon<Float>));

        Mask sample_diffuse = active && sample1 < prob_diffuse;
        Mask sample_fluoro = active && !sample_diffuse;

        bs.wo = warp::square_to_cosine_hemisphere(sample2);
        bs.pdf = warp::square_to_cosine_hemisphere_pdf(bs.wo);
        bs.eta = 1.f;

        UnpolarizedSpectrum result(0.f);

        if (dr::any_or<true>(sample_diffuse)) {
            dr::masked(bs.sampled_component, sample_diffuse) = 0;
            dr::masked(bs.sampled_type, sample_diffuse) =
                +BSDFFlags::DiffuseReflection;
            dr::masked(bs.pdf, sample_diffuse) *= prob_diffuse;
            result[sample_diffuse] = diffuse_value / prob_diffuse;
        }
        if (dr::any_or<true>(sample_fluoro)) {
            dr::masked(bs.sampled_component, sample_fluoro) = 1;
            dr::masked(bs.sampled_type, sample_fluoro) =
                +BSDFFlags::FluorescentReflection;
            dr::masked(bs.pdf, sample_fluoro) *= (1.f - prob_diffuse);
            result[sample_fluoro] = fluoro_value / (1.f - prob_diffuse);
        }

        return { bs, depolarizer<Spectrum>(result) & (active && bs.pdf > 0.f) };
    }

    std::pair<Wavelength, UnpolarizedSpectrum> sample_wavelength_shift(
        const BSDFContext &ctx, const SurfaceInteraction3f &si,
        Float sample, Mask active) const override {
        auto spectrum = ctx.mode == TransportMode::Radiance ? m_excitation : m_fluorescence;
        auto [wavelengths, weight] = spectrum->sample_spectrum(
            si, math::sample_shifted<Wavelength>(sample), active);

        // The emission spectrum should have integral normalised to 1
        return { wavelengths,
                 ctx.mode == TransportMode::Radiance ? weight : weight / spectrum->sum(si, active) };
    };

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        bool has_diffuse = ctx.is_enabled(BSDFFlags::DiffuseReflection, 0);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        if (unlikely(!has_diffuse || dr::none_or<false>(active)))
            return { 0.f };

        UnpolarizedSpectrum value =
            m_reflectance->eval(si, active) * dr::InvPi<Float> * cos_theta_o;

        return depolarizer<Spectrum>(value) & active;
    }

    Spectrum eval_fluoro(const BSDFContext &ctx,
                         const SurfaceInteraction3f &si,
                         const Vector3f &wo,
                         Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        bool has_fluoro = ctx.is_enabled(BSDFFlags::FluorescentReflection, 1);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        if (unlikely(!has_fluoro || dr::none_or<false>(active)))
            return { 0.f };

        UnpolarizedSpectrum fluoro_value = ctx.mode == TransportMode::Radiance
            ? m_fluorescence->eval_norm(si, active)
            : m_excitation->eval(si, active);

        UnpolarizedSpectrum value = fluoro_value * dr::InvPi<Float> * cos_theta_o;

        return depolarizer<Spectrum>(value) & active;
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        bool has_diffuse  = ctx.is_enabled(BSDFFlags::DiffuseReflection, 0),
             has_fluoro = ctx.is_enabled(BSDFFlags::FluorescentReflection, 1);

        if (!has_diffuse && !has_fluoro)
            return 0.f;

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        Float pdf = warp::square_to_cosine_hemisphere_pdf(wo);

        UnpolarizedSpectrum diffuse_value = m_reflectance->eval(si, active);
        UnpolarizedSpectrum fluoro_value = ctx.mode == TransportMode::Radiance
            ? m_fluorescence->eval_norm(si, active)
            : m_excitation->eval(si, active);

        Float prob_diffuse = 1.f;
        Float fluoro_scale = ctx.mode == TransportMode::Radiance
            ? m_excitation->sum(si, active) : 1.f;
        if (unlikely(has_fluoro != has_diffuse))
            prob_diffuse = has_diffuse ? 1.f : 0.f;
        else
            // TODO: bad approximation (?) when the number of wavelengths per ray > 1
            prob_diffuse = dr::mean(
                diffuse_value /
                (diffuse_value + fluoro_scale * fluoro_value + dr::Epsilon<Float>));

        return dr::select(cos_theta_i > 0.f && cos_theta_o > 0.f, pdf * prob_diffuse, 0.f);
    }

    std::pair<Spectrum, Float> eval_pdf(const BSDFContext &ctx,
                                        const SurfaceInteraction3f &si,
                                        const Vector3f &wo,
                                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        bool has_diffuse  = ctx.is_enabled(BSDFFlags::DiffuseReflection, 0),
             has_fluoro = ctx.is_enabled(BSDFFlags::FluorescentReflection, 1);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        if (unlikely(!has_diffuse || dr::none_or<false>(active)))
            return { 0.f, 0.f };

        UnpolarizedSpectrum diffuse_value = m_reflectance->eval(si, active);
        UnpolarizedSpectrum fluoro_value = ctx.mode == TransportMode::Radiance
            ? m_fluorescence->eval_norm(si, active)
            : m_excitation->eval(si, active);

        Float prob_diffuse = 1.f;
        Float fluoro_scale = ctx.mode == TransportMode::Radiance
            ? m_excitation->sum(si, active) : 1.f;
        if (unlikely(has_fluoro != has_diffuse))
            prob_diffuse = has_diffuse ? 1.f : 0.f;
        else
            // TODO: bad approximation (?) when the number of wavelengths per ray > 1
            prob_diffuse = dr::mean(
                diffuse_value /
                (diffuse_value + fluoro_scale * fluoro_value + dr::Epsilon<Float>));

        UnpolarizedSpectrum value = diffuse_value * dr::InvPi<Float> * cos_theta_o;

        Float pdf = warp::square_to_cosine_hemisphere_pdf(wo) / prob_diffuse;

        return { depolarizer<Spectrum>(value) & active, dr::select(active, pdf, 0.f) };
    }

    std::pair<Spectrum, Float> eval_pdf_fluoro(const BSDFContext &ctx,
                                               const SurfaceInteraction3f &si,
                                               const Vector3f &wo,
                                               Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        bool has_diffuse  = ctx.is_enabled(BSDFFlags::DiffuseReflection, 0),
             has_fluoro = ctx.is_enabled(BSDFFlags::FluorescentReflection, 1);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        if (unlikely(!has_fluoro || dr::none_or<false>(active)))
            return { 0.f, 0.f };

        UnpolarizedSpectrum diffuse_value = m_reflectance->eval(si, active);
        UnpolarizedSpectrum fluoro_value = ctx.mode == TransportMode::Radiance
            ? m_fluorescence->eval_norm(si, active)
            : m_excitation->eval(si, active);

        Float prob_diffuse = 1.f;
        Float fluoro_scale = ctx.mode == TransportMode::Radiance
            ? m_excitation->sum(si, active) : 1.f;
        if (unlikely(has_fluoro != has_diffuse))
            prob_diffuse = has_diffuse ? 1.f : 0.f;
        else
            // TODO: bad approximation (?) when the number of wavelengths per ray > 1
            prob_diffuse = dr::mean(
                diffuse_value /
                (diffuse_value + fluoro_scale * fluoro_value + dr::Epsilon<Float>));

        UnpolarizedSpectrum value = fluoro_value * dr::InvPi<Float> * cos_theta_o;

        Float pdf = warp::square_to_cosine_hemisphere_pdf(wo) / (1.f - prob_diffuse);

        return { depolarizer<Spectrum>(value) & active, dr::select(active, pdf, 0.f) };
    }

    Spectrum eval_diffuse_reflectance(const SurfaceInteraction3f &si,
                                      Mask active) const override {
        return m_reflectance->eval(si, active);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SmoothDiffuseFluoro[" << std::endl
            << "  reflectance = "  << string::indent(m_reflectance)  << "," << std::endl
            << "  fluorescence = " << string::indent(m_fluorescence) << "," << std::endl
            << "  excitation = "   << string::indent(m_excitation)   << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(SmoothDiffuseFluoro)
private:
    ref<Texture> m_reflectance;
    ref<Texture> m_fluorescence;
    ref<Texture> m_excitation;

    MI_TRAVERSE_CB(Base, m_reflectance, m_fluorescence, m_excitation)
};

MI_EXPORT_PLUGIN(SmoothDiffuseFluoro)
NAMESPACE_END(mitsuba)
