#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _bsdf-diffuse:

Smooth diffuse material (:monosp:`diffuse`)
-------------------------------------------

.. pluginparameters::

 * - reflectance
   - |spectrum| or |texture|
   - Specifies the diffuse albedo of the material (Default: 0.5)
   - |exposed|, |differentiable|

The smooth diffuse material (also referred to as *Lambertian*)
represents an ideally diffuse material with a user-specified amount of
reflectance. Any received illumination is scattered so that the surface
looks the same independently of the direction of observation.

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/bsdf_diffuse_plain.jpg
   :caption: Homogeneous reflectance
.. subfigure:: ../../resources/data/docs/images/render/bsdf_diffuse_textured.jpg
   :caption: Textured reflectance
.. subfigend::
   :label: fig-diffuse

Apart from a homogeneous reflectance value, the plugin can also accept
a nested or referenced texture map to be used as the source of reflectance
information, which is then mapped onto the shape based on its UV
parameterization. When no parameters are specified, the model uses the default
of 50% reflectance.

Note that this material is one-sided---that is, observed from the
back side, it will be completely black. If this is undesirable,
consider using the :ref:`twosided <bsdf-twosided>` BRDF adapter plugin.
The following XML snippet describes a diffuse material,
whose reflectance is specified as an sRGB color:

.. tabs::
    .. code-tab:: xml
        :name: diffuse-srgb

        <bsdf type="diffuse">
            <rgb name="reflectance" value="0.2, 0.25, 0.7"/>
        </bsdf>

    .. code-tab:: python

        'type': 'diffuse',
        'reflectance': {
            'type': 'rgb',
            'value': [0.2, 0.25, 0.7]
        }

Alternatively, the reflectance can be textured:

.. tabs::
    .. code-tab:: xml
        :name: diffuse-texture

        <bsdf type="diffuse">
            <texture type="bitmap" name="reflectance">
                <string name="filename" value="wood.jpg"/>
            </texture>
        </bsdf>

    .. code-tab:: python

        'type': 'diffuse',
        'reflectance': {
            'type': 'bitmap',
            'filename': 'wood.jpg'
        }
*/
template <typename Float, typename Spectrum>
class SmoothDiffuseFluoro final : public BSDF<Float, Spectrum> {
public:
    MI_IMPORT_BASE(BSDF, m_flags, m_components)
    MI_IMPORT_TYPES(Texture)

    SmoothDiffuseFluoro(const Properties &props) : Base(props) {
        m_reflectance = props.get_texture<Texture>("reflectance", .5f);
        m_fluorescence = props.get_texture<Texture>("fluorescence", .5f);
        m_excitation = props.get_texture<Texture>("excitation", 0.0f);

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
        
        // Ignoring perfectly grazing incoming rays
        active &= cos_theta_i > 0.f;

        if (unlikely((!has_diffuse && !has_fluoro) || dr::none_or<false>(active)))
            return { bs, 0.f };

        Float prob_diffuse = 0.0; // si.wavelengths[0] * m_excitation->eval(si, active);
        Mask sample_diffuse = active && sample1 < prob_diffuse;
        Mask sample_fluoro = active && !sample_diffuse;
        
        bs.wo = warp::square_to_cosine_hemisphere(sample2);
        bs.pdf = warp::square_to_cosine_hemisphere_pdf(bs.wo);
        bs.eta = 1.f;

        if (dr::any_or<true>(sample_diffuse)) {
            dr::masked(bs.sampled_component, sample_diffuse) = 0;
            dr::masked(bs.sampled_type, sample_diffuse) =
                    +BSDFFlags::DiffuseReflection;
            active &= !sample_diffuse;
        }
        if (dr::any_or<true>(sample_fluoro)) {
            dr::masked(bs.sampled_component, sample_fluoro) = 1;
            dr::masked(bs.sampled_type, sample_fluoro) =
                    +BSDFFlags::FluorescentReflection;
            active &= !sample_fluoro;
        }

        UnpolarizedSpectrum value(0.f);
        if (has_diffuse) {
            value += m_reflectance->eval(si, active);
        }
        if (has_fluoro) {
            value += m_fluorescence->eval(si, active);
        }

        return { bs, depolarizer<Spectrum>(value) & (active && bs.pdf > 0.f) };
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        bool has_diffuse  = ctx.is_enabled(BSDFFlags::DiffuseReflection, 0),
             has_fluoro = ctx.is_enabled(BSDFFlags::FluorescentReflection, 1);
        
        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);
        
        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        if (unlikely((!has_diffuse && !has_fluoro) || dr::none_or<false>(active)))
            return { 0.f };
        
        UnpolarizedSpectrum value(0.f);
        if (has_diffuse) {
            value += m_reflectance->eval(si, active) * dr::InvPi<Float> * cos_theta_o;
        }
        if (has_fluoro) {
            value += m_fluorescence->eval(si, active) * dr::InvPi<Float> * cos_theta_o;
        }

        return depolarizer<Spectrum>(value) & active;
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        if (!ctx.is_enabled(BSDFFlags::DiffuseReflection) && 
            !ctx.is_enabled(BSDFFlags::FluorescentReflection))
            return 0.f;

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        Float pdf = warp::square_to_cosine_hemisphere_pdf(wo);

        return dr::select(cos_theta_i > 0.f && cos_theta_o > 0.f, pdf, 0.f);
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

        if (unlikely((!has_diffuse && !has_fluoro) || dr::none_or<false>(active)))
            return { 0.f, 0.f };
        
        UnpolarizedSpectrum value(0.f);
        if (has_diffuse) {
            value += m_reflectance->eval(si, active) * dr::InvPi<Float> * cos_theta_o;
        }
        if (has_fluoro) {
            value += m_fluorescence->eval(si, active) * dr::InvPi<Float> * cos_theta_o;
        }

        Float pdf = warp::square_to_cosine_hemisphere_pdf(wo);

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
